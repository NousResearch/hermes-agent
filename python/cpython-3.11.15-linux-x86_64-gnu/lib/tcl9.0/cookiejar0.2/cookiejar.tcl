# cookiejar.tcl --
#
#	Implementation of an HTTP cookie storage engine using SQLite. The
#	implementation is done as a TclOO class, and includes a punycode
#	encoder and decoder (though only the encoder is currently used).
#
# See the file "license.terms" for information on usage and redistribution of
# this file, and for a DISCLAIMER OF ALL WARRANTIES.

# Dependencies
package require Tcl 8.6-
package require http 2.8.4
package require sqlite3
package require tcl::idna 1.0

#
# Configuration for the cookiejar package, plus basic support procedures.
#

# This is the class that we are creating
if {![llength [info commands ::http::cookiejar]]} {
    ::oo::class create ::http::cookiejar
}

namespace eval [info object namespace ::http::cookiejar] {
    proc setInt {*var val} {
	upvar 1 ${*var} var
	if {[catch {incr dummy $val} msg]} {
	    return -code error $msg
	}
	set var $val
    }
    proc setInterval {trigger *var val} {
	upvar 1 ${*var} var
	if {![string is integer -strict $val] || $val < 1} {
	    return -code error "expected positive integer but got \"$val\""
	}
	set var $val
	{*}$trigger
    }
    proc setBool {*var val} {
	upvar 1 ${*var} var
	if {[catch {if {$val} {}} msg]} {
	    return -code error $msg
	}
	set var [expr {!!$val}]
    }

    proc setLog {*var val} {
	upvar 1 ${*var} var
	set var [::tcl::prefix match -message "log level" \
		{debug info warn error} $val]
    }

    # Keep this in sync with pkgIndex.tcl and with the install directories in
    # Makefiles
    variable version 0.2.0

    variable domainlist \
	https://publicsuffix.org/list/public_suffix_list.dat
    variable domainfile \
	[file join [file dirname [info script]] public_suffix_list.dat.gz]
    # The list is directed to from http://publicsuffix.org/list/
    variable loglevel info
    variable vacuumtrigger 200
    variable retainlimit 100
    variable offline false
    variable purgeinterval 60000
    variable refreshinterval 10000000
    variable domaincache {}

    # Some support procedures, none particularly useful in general
    namespace eval support {
	# Set up a logger if the http package isn't actually loaded yet.
	if {![llength [info commands ::http::Log]]} {
	    proc ::http::Log args {
		# Do nothing by default...
	    }
	}

	namespace export *
	proc locn {secure domain path {key ""}} {
	    if {$key eq ""} {
		format "%s://%s%s" [expr {$secure?"https":"http"}] \
		    [::tcl::idna encode $domain] $path
	    } else {
		format "%s://%s%s?%s" \
		    [expr {$secure?"https":"http"}] [::tcl::idna encode $domain] \
		    $path $key
	    }
	}
	proc splitDomain domain {
	    set pieces [split $domain "."]
	    for {set i [llength $pieces]} {[incr i -1] >= 0} {} {
		lappend result [join [lrange $pieces $i end] "."]
	    }
	    return $result
	}
	proc splitPath path {
	    set pieces [split [string trimleft $path "/"] "/"]
	    set result /
	    for {set j 0} {$j < [llength $pieces]} {incr j} {
		lappend result /[join [lrange $pieces 0 $j] "/"]
	    }
	    return $result
	}
	proc isoNow {} {
	    set ms [clock milliseconds]
	    set ts [expr {$ms / 1000}]
	    set ms [format %03d [expr {$ms % 1000}]]
	    clock format $ts -format "%Y%m%dT%H%M%S.${ms}Z" -gmt 1
	}
	proc log {level msg args} {
	    namespace upvar [info object namespace ::http::cookiejar] \
		loglevel loglevel
	    set who [uplevel 1 self class]
	    set mth [uplevel 1 self method]
	    set map {debug 0 info 1 warn 2 error 3}
	    if {[string map $map $level] >= [string map $map $loglevel]} {
		set msg [format $msg {*}$args]
		set LVL [string toupper $level]
		::http::Log "[isoNow] $LVL $who $mth - $msg"
	    }
	}
    }
}

# Now we have enough information to provide the package.
package provide cookiejar \
    [set [info object namespace ::http::cookiejar]::version]

# The implementation of the cookiejar package
::oo::define ::http::cookiejar {
    self {
	method configure {{optionName "\x00\x00"} {optionValue "\x00\x00"}} {
	    set tbl {
		-domainfile    {domainfile set}
		-domainlist    {domainlist set}
		-domainrefresh {refreshinterval setInterval}
		-loglevel      {loglevel setLog}
		-offline       {offline setBool}
		-purgeold      {purgeinterval setInterval}
		-retain        {retainlimit setInt}
		-vacuumtrigger {vacuumtrigger setInt}
	    }
	    dict lappend tbl -domainrefresh [namespace code {
		my IntervalTrigger PostponeRefresh
	    }]
	    dict lappend tbl -purgeold [namespace code {
		my IntervalTrigger PostponePurge
	    }]
	    if {$optionName eq "\x00\x00"} {
		return [dict keys $tbl]
	    }
	    set opt [::tcl::prefix match -message "option" \
		    [dict keys $tbl] $optionName]
	    set setter [lassign [dict get $tbl $opt] varname]
	    namespace upvar [namespace current] $varname var
	    if {$optionValue ne "\x00\x00"} {
		{*}$setter var $optionValue
	    }
	    return $var
	}

	method IntervalTrigger {method} {
	    # TODO: handle subclassing
	    foreach obj [info class instances [self]] {
		[info object namespace $obj]::my $method
	    }
	}
    }

    variable purgeTimer deletions refreshTimer
    constructor {{path ""}} {
	namespace import [info object namespace [self class]]::support::*

	if {$path eq ""} {
	    sqlite3 [namespace current]::db :memory:
	    set storeorigin "constructed cookie store in memory"
	} else {
	    sqlite3 [namespace current]::db $path
	    db timeout 500
	    set storeorigin "loaded cookie store from $path"
	}

	set deletions 0
	db transaction {
	    db eval {
		--;# Store the persistent cookies in this table.
		--;# Deletion policy: once they expire, or if explicitly
		--;# killed.
		CREATE TABLE IF NOT EXISTS persistentCookies (
		    id INTEGER PRIMARY KEY,
		    secure INTEGER NOT NULL,
		    domain TEXT NOT NULL COLLATE NOCASE,
		    path TEXT NOT NULL,
		    key TEXT NOT NULL,
		    value TEXT NOT NULL,
		    originonly INTEGER NOT NULL,
		    expiry INTEGER NOT NULL,
		    lastuse INTEGER NOT NULL,
		    creation INTEGER NOT NULL);
		CREATE UNIQUE INDEX IF NOT EXISTS persistentUnique
		    ON persistentCookies (domain, path, key);
		CREATE INDEX IF NOT EXISTS persistentLookup
		    ON persistentCookies (domain, path);

		--;# Store the session cookies in this table.
		--;# Deletion policy: at cookiejar instance deletion, if
		--;# explicitly killed, or if the number of session cookies is
		--;# too large and the cookie has not been used recently.
		CREATE TEMP TABLE sessionCookies (
		    id INTEGER PRIMARY KEY,
		    secure INTEGER NOT NULL,
		    domain TEXT NOT NULL COLLATE NOCASE,
		    path TEXT NOT NULL,
		    key TEXT NOT NULL,
		    originonly INTEGER NOT NULL,
		    value TEXT NOT NULL,
		    lastuse INTEGER NOT NULL,
		    creation INTEGER NOT NULL);
		CREATE UNIQUE INDEX sessionUnique
		    ON sessionCookies (domain, path, key);
		CREATE INDEX sessionLookup ON sessionCookies (domain, path);

		--;# View to allow for simple looking up of a cookie.
		--;# Deletion policy: NOT SUPPORTED via this view.
		CREATE TEMP VIEW cookies AS
		    SELECT id, domain, (
			    CASE originonly WHEN 1 THEN path ELSE '.' || path END
			) AS path, key, value, secure, 1 AS persistent
			FROM persistentCookies
		    UNION
		    SELECT id, domain, (
			    CASE originonly WHEN 1 THEN path ELSE '.' || path END
			) AS path, key, value, secure, 0 AS persistent
			FROM sessionCookies;

		--;# Encoded domain permission policy; if forbidden is 1, no
		--;# cookie may be ever set for the domain, and if forbidden
		--;# is 0, cookies *may* be created for the domain (overriding
		--;# the forbiddenSuper table).
		--;# Deletion policy: normally not modified.
		CREATE TABLE IF NOT EXISTS domains (
		    domain TEXT PRIMARY KEY NOT NULL,
		    forbidden INTEGER NOT NULL);

		--;# Domains that may not have a cookie defined for direct
		--;# child domains of them.
		--;# Deletion policy: normally not modified.
		CREATE TABLE IF NOT EXISTS forbiddenSuper (
		    domain TEXT PRIMARY KEY);

		--;# When we last retrieved the domain list.
		CREATE TABLE IF NOT EXISTS domainCacheMetadata (
		    id INTEGER PRIMARY KEY,
		    retrievalDate INTEGER,
		    installDate INTEGER);
	    }

	    set cookieCount "no"
	    db eval {
		SELECT COUNT(*) AS cookieCount FROM persistentCookies
	    }
	    log info "%s with %s entries" $storeorigin $cookieCount

	    my PostponePurge

	    if {$path ne ""} {
		if {[db exists {SELECT 1 FROM domains}]} {
		    my RefreshDomains
		} else {
		    my InitDomainList
		    my PostponeRefresh
		}
	    } else {
		set data [my GetDomainListOffline metadata]
		my InstallDomainData $data $metadata
		my PostponeRefresh
	    }
	}
    }

    method PostponePurge {} {
	namespace upvar [info object namespace [self class]] \
	    purgeinterval interval
	catch {after cancel $purgeTimer}
	set purgeTimer [after $interval [namespace code {my PurgeCookies}]]
    }

    method PostponeRefresh {} {
	namespace upvar [info object namespace [self class]] \
	    refreshinterval interval
	catch {after cancel $refreshTimer}
	set refreshTimer [after $interval [namespace code {my RefreshDomains}]]
    }

    method RefreshDomains {} {
	# TODO: domain list refresh policy
	my PostponeRefresh
    }

    method HttpGet {url {timeout 0} {maxRedirects 5}} {
	for {set r 0} {$r < $maxRedirects} {incr r} {
	    set tok [::http::geturl $url -timeout $timeout]
	    try {
		if {[::http::status $tok] eq "timeout"} {
		    return -code error "connection timed out"
		} elseif {[::http::ncode $tok] == 200} {
		    return [::http::data $tok]
		} elseif {[::http::ncode $tok] >= 400} {
		    return -code error [::http::error $tok]
		} elseif {[dict exists [::http::meta $tok] Location]} {
		    set url [dict get [::http::meta $tok] Location]
		    continue
		}
		return -code error \
		    "unexpected state: [::http::code $tok]"
	    } finally {
		::http::cleanup $tok
	    }
	}
	return -code error "too many redirects"
    }
    method GetDomainListOnline {metaVar} {
	upvar 1 $metaVar meta
	namespace upvar [info object namespace [self class]] \
	    domainlist url domaincache cache
	lassign $cache when data
	if {$when > [clock seconds] - 3600} {
	    log debug "using cached value created at %s" \
		[clock format $when -format {%Y%m%dT%H%M%SZ} -gmt 1]
	    dict set meta retrievalDate $when
	    return $data
	}
	log debug "loading domain list from %s" $url
	try {
	    set when [clock seconds]
	    set data [my HttpGet $url]
	    set cache [list $when $data]
	    # TODO: Should we use the Last-Modified header instead?
	    dict set meta retrievalDate $when
	    return $data
	} on error msg {
	    log error "failed to fetch list of forbidden cookie domains from %s: %s" \
		    $url $msg
	    return {}
	}
    }
    method GetDomainListOffline {metaVar} {
	upvar 1 $metaVar meta
	namespace upvar [info object namespace [self class]] \
	    domainfile filename
	log debug "loading domain list from %s" $filename
	try {
	    set f [open $filename]
	    try {
		if {[string match *.gz $filename]} {
		    zlib push gunzip $f
		}
		fconfigure $f -encoding utf-8
		dict set meta retrievalDate [file mtime $filename]
		return [read $f]
	    } finally {
		close $f
	    }
	} on error {msg opt} {
	    log error "failed to read list of forbidden cookie domains from %s: %s" \
		    $filename $msg
	    return -options $opt $msg
	}
    }
    method InitDomainList {} {
	namespace upvar [info object namespace [self class]] \
	    offline offline
	if {!$offline} {
	    try {
		set data [my GetDomainListOnline metadata]
		if {[string length $data]} {
		    my InstallDomainData $data $metadata
		    return
		}
	    } on error {} {
		log warn "attempting to fall back to built in version"
	    }
	}
	set data [my GetDomainListOffline metadata]
	my InstallDomainData $data $metadata
    }

    method InstallDomainData {data meta} {
	set n [db total_changes]
	db transaction {
	    foreach line [split $data "\n"] {
		if {[string trim $line] eq ""} {
		    continue
		} elseif {[string match //* $line]} {
		    continue
		} elseif {[string match !* $line]} {
		    set line [string range $line 1 end]
		    set idna [string tolower [::tcl::idna encode $line]]
		    set utf [::tcl::idna decode [string tolower $line]]
		    db eval {
			INSERT OR REPLACE INTO domains (domain, forbidden)
			VALUES ($utf, 0);
		    }
		    if {$idna ne $utf} {
			db eval {
			    INSERT OR REPLACE INTO domains (domain, forbidden)
			    VALUES ($idna, 0);
			}
		    }
		} else {
		    if {[string match {\*.*} $line]} {
			set line [string range $line 2 end]
			set idna [string tolower [::tcl::idna encode $line]]
			set utf [::tcl::idna decode [string tolower $line]]
			db eval {
			    INSERT OR REPLACE INTO forbiddenSuper (domain)
			    VALUES ($utf);
			}
			if {$idna ne $utf} {
			    db eval {
				INSERT OR REPLACE INTO forbiddenSuper (domain)
				VALUES ($idna);
			    }
			}
		    } else {
			set idna [string tolower [::tcl::idna encode $line]]
			set utf [::tcl::idna decode [string tolower $line]]
		    }
		    db eval {
			INSERT OR REPLACE INTO domains (domain, forbidden)
			VALUES ($utf, 1);
		    }
		    if {$idna ne $utf} {
			db eval {
			    INSERT OR REPLACE INTO domains (domain, forbidden)
			    VALUES ($idna, 1);
			}
		    }
		}
		if {$utf ne [::tcl::idna decode [string tolower $idna]]} {
		    log warn "mismatch in IDNA handling for %s (%d, %s, %s)" \
			    $idna $line $utf [::tcl::idna decode $idna]
		}
	    }

	    dict with meta {
		set installDate [clock seconds]
		db eval {
		    INSERT OR REPLACE INTO domainCacheMetadata
			(id, retrievalDate, installDate)
		    VALUES (1, $retrievalDate, $installDate);
		}
	    }
	}
	set n [expr {[db total_changes] - $n}]
	log info "constructed domain info with %d entries" $n
    }

    # This forces the rebuild of the domain data, loading it from
    method forceLoadDomainData {} {
	db transaction {
	    db eval {
		DELETE FROM domains;
		DELETE FROM forbiddenSuper;
		INSERT OR REPLACE INTO domainCacheMetadata
		    (id, retrievalDate, installDate)
		VALUES (1, -1, -1);
	    }
	    my InitDomainList
	}
    }

    destructor {
	catch {
	    after cancel $purgeTimer
	}
	catch {
	    after cancel $refreshTimer
	}
	catch {
	    db close
	}
	return
    }

    method GetCookiesForHostAndPath {listVar secure host path fullhost} {
	upvar 1 $listVar result
	log debug "check for cookies for %s" [locn $secure $host $path]
	set exact [expr {$host eq $fullhost}]
	db eval {
	    SELECT key, value FROM persistentCookies
	    WHERE domain = $host AND path = $path AND secure <= $secure
		AND (NOT originonly OR domain = $fullhost)
		AND originonly = $exact
	} {
	    lappend result $key $value
	    db eval {
		UPDATE persistentCookies SET lastuse = $now WHERE id = $id
	    }
	}
	set now [clock seconds]
	db eval {
	    SELECT id, key, value FROM sessionCookies
	    WHERE domain = $host AND path = $path AND secure <= $secure
		AND (NOT originonly OR domain = $fullhost)
		AND originonly = $exact
	} {
	    lappend result $key $value
	    db eval {
		UPDATE sessionCookies SET lastuse = $now WHERE id = $id
	    }
	}
    }

    method getCookies {proto host path} {
	set result {}
	set paths [splitPath $path]
	if {[regexp {[^0-9.]} $host]} {
	    set domains [splitDomain [string tolower [::tcl::idna encode $host]]]
	} else {
	    # Ugh, it's a numeric domain! Restrict it to just itself...
	    set domains [list $host]
	}
	set secure [string equal -nocase $proto "https"]
	# Open question: how to move these manipulations into the database
	# engine (if that's where they *should* be).
	#
	# Suggestion from kbk:
	#LENGTH(theColumn) <= LENGTH($queryStr) AND
	#SUBSTR(theColumn, LENGTH($queryStr) LENGTH(theColumn)+1) = $queryStr
	#
	# However, we instead do most of the work in Tcl because that lets us
	# do the splitting exactly right, and it's far easier to work with
	# strings in Tcl than in SQL.
	db transaction {
	    foreach domain $domains {
		foreach p $paths {
		    my GetCookiesForHostAndPath result $secure $domain $p $host
		}
	    }
	    return $result
	}
    }

    method BadDomain options {
	if {![dict exists $options domain]} {
	    log error "no domain present in options"
	    return 0
	}
	dict with options {}
	if {$domain ne $origin} {
	    log debug "cookie domain varies from origin (%s, %s)" \
		    $domain $origin
	    if {[string match .* $domain]} {
		set dotd $domain
	    } else {
		set dotd .$domain
	    }
	    if {![string equal -length [string length $dotd] \
		    [string reverse $dotd] [string reverse $origin]]} {
		log warn "bad cookie: domain not suffix of origin"
		return 1
	    }
	}
	if {![regexp {[^0-9.]} $domain]} {
	    if {$domain eq $origin} {
		# May set for itself
		return 0
	    }
	    log warn "bad cookie: for a numeric address"
	    return 1
	}
	db eval {
	    SELECT forbidden FROM domains WHERE domain = $domain
	} {
	    if {$forbidden} {
		log warn "bad cookie: for a forbidden address"
	    }
	    return $forbidden
	}
	if {[regexp {^[^.]+\.(.+)$} $domain -> super] && [db exists {
	    SELECT 1 FROM forbiddenSuper WHERE domain = $super
	}]} then {
	    log warn "bad cookie: for a forbidden address"
	    return 1
	}
	return 0
    }

    # A defined extension point to allow users to easily impose extra policies
    # on whether to accept cookies from a particular domain and path.
    method policyAllow {operation domain path} {
	return true
    }

    method storeCookie {options} {
	db transaction {
	    if {[my BadDomain $options]} {
		return
	    }
	    set now [clock seconds]
	    set persistent [dict exists $options expires]
	    dict with options {}
	    if {!$persistent} {
		if {![my policyAllow session $domain $path]} {
		    log warn "bad cookie: $domain prohibited by user policy"
		    return
		}
		db eval {
		    INSERT OR REPLACE INTO sessionCookies (
			secure, domain, path, key, value, originonly, creation,
			lastuse)
		    VALUES ($secure, $domain, $path, $key, $value, $hostonly,
			$now, $now);
		    DELETE FROM persistentCookies
		    WHERE domain = $domain AND path = $path AND key = $key
			AND secure <= $secure AND originonly = $hostonly
		}
		incr deletions [db changes]
		log debug "defined session cookie for %s" \
			[locn $secure $domain $path $key]
	    } elseif {$expires < $now} {
		if {![my policyAllow delete $domain $path]} {
		    log warn "bad cookie: $domain prohibited by user policy"
		    return
		}
		db eval {
		    DELETE FROM persistentCookies
		    WHERE domain = $domain AND path = $path AND key = $key
			AND secure <= $secure AND originonly = $hostonly
		}
		set del [db changes]
		db eval {
		    DELETE FROM sessionCookies
		    WHERE domain = $domain AND path = $path AND key = $key
			AND secure <= $secure AND originonly = $hostonly
		}
		incr deletions [incr del [db changes]]
		log debug "deleted %d cookies for %s" \
			$del [locn $secure $domain $path $key]
	    } else {
		if {![my policyAllow set $domain $path]} {
		    log warn "bad cookie: $domain prohibited by user policy"
		    return
		}
		db eval {
		    INSERT OR REPLACE INTO persistentCookies (
			secure, domain, path, key, value, originonly, expiry,
			creation, lastuse)
		    VALUES ($secure, $domain, $path, $key, $value, $hostonly,
			$expires, $now, $now);
		    DELETE FROM sessionCookies
		    WHERE domain = $domain AND path = $path AND key = $key
			AND secure <= $secure AND originonly = $hostonly
		}
		incr deletions [db changes]
		log debug "defined persistent cookie for %s, expires at %s" \
			[locn $secure $domain $path $key] \
			[clock format $expires]
	    }
	}
    }

    method PurgeCookies {} {
	namespace upvar [info object namespace [self class]] \
	    vacuumtrigger trigger  retainlimit retain
	my PostponePurge
	set now [clock seconds]
	log debug "purging cookies that expired before %s" [clock format $now]
	db transaction {
	    db eval {
		DELETE FROM persistentCookies WHERE expiry < $now
	    }
	    incr deletions [db changes]
	    db eval {
		DELETE FROM persistentCookies WHERE id IN (
		    SELECT id FROM persistentCookies ORDER BY lastuse ASC
		    LIMIT -1 OFFSET $retain)
	    }
	    incr deletions [db changes]
	    db eval {
		DELETE FROM sessionCookies WHERE id IN (
		    SELECT id FROM sessionCookies ORDER BY lastuse
		    LIMIT -1 OFFSET $retain)
	    }
	    incr deletions [db changes]
	}

	# Once we've deleted a fair bit, vacuum the database. Must be done
	# outside a transaction.
	if {$deletions > $trigger} {
	    set deletions 0
	    log debug "vacuuming cookie database"
	    catch {
		db eval {
		    VACUUM
		}
	    }
	}
    }

    forward Database db

    method lookup {{host ""} {key ""}} {
	set host [string tolower [::tcl::idna encode $host]]
	db transaction {
	    if {$host eq ""} {
		set result {}
		db eval {
		    SELECT DISTINCT domain FROM cookies
		    ORDER BY domain
		} {
		    lappend result [::tcl::idna decode [string tolower $domain]]
		}
		return $result
	    } elseif {$key eq ""} {
		set result {}
		db eval {
		    SELECT DISTINCT key FROM cookies
		    WHERE domain = $host
		    ORDER BY key
		} {
		    lappend result $key
		}
		return $result
	    } else {
		db eval {
		    SELECT value FROM cookies
		    WHERE domain = $host AND key = $key
		    LIMIT 1
		} {
		    return $value
		}
		return -code error "no such key for that host"
	    }
	}
    }
}

# Local variables:
# mode: tcl
# fill-column: 78
# End:
