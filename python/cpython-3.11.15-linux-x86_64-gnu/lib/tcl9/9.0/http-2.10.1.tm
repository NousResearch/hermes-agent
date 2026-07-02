# http.tcl --
#
#	Client-side HTTP for GET, POST, and HEAD commands. These routines can
#	be used in untrusted code that uses the Safesock security policy.
#	These procedures use a callback interface to avoid using vwait, which
#	is not defined in the safe base.
#
# See the file "license.terms" for information on usage and redistribution of
# this file, and for a DISCLAIMER OF ALL WARRANTIES.

package require Tcl 8.6-
# Keep this in sync with pkgIndex.tcl and with the install directories in
# Makefiles
package provide http 2.10.1

namespace eval http {
    # Allow resourcing to not clobber existing data

    variable http
    if {![info exists http]} {
	array set http {
	    -accept */*
	    -cookiejar {}
	    -pipeline 1
	    -postfresh 0
	    -proxyhost {}
	    -proxyport {}
	    -proxyfilter http::ProxyRequired
	    -proxynot {}
	    -proxyauth {}
	    -repost 0
	    -threadlevel 0
	    -urlencoding utf-8
	    -zip 1
	}
	# We need a useragent string of this style or various servers will
	# refuse to send us compressed content even when we ask for it. This
	# follows the de-facto layout of user-agent strings in current browsers.
	# Safe interpreters do not have ::tcl_platform(os) or
	# ::tcl_platform(osVersion).
	if {[interp issafe]} {
	    set http(-useragent) "Mozilla/5.0\
		(Windows; U;\
		Windows NT 10.0)\
		http/[package provide http] Tcl/[package provide Tcl]"
	} else {
	    set http(-useragent) "Mozilla/5.0\
		([string totitle $::tcl_platform(platform)]; U;\
		$::tcl_platform(os) $::tcl_platform(osVersion))\
		http/[package provide http] Tcl/[package provide Tcl]"
	}
    }

    proc init {} {
	# Set up the map for quoting chars. RFC3986 Section 2.3 say percent
	# encode all except: "... percent-encoded octets in the ranges of
	# ALPHA (%41-%5A and %61-%7A), DIGIT (%30-%39), hyphen (%2D), period
	# (%2E), underscore (%5F), or tilde (%7E) should not be created by URI
	# producers ..."
	for {set i 0} {$i <= 256} {incr i} {
	    set c [format %c $i]
	    if {![string match {[-._~a-zA-Z0-9]} $c]} {
		set map($c) %[format %.2X $i]
	    }
	}
	# These are handled specially
	set map(\n) %0D%0A
	variable formMap [array get map]

	# Create a map for HTTP/1.1 open sockets
	variable socketMapping
	variable socketRdState
	variable socketWrState
	variable socketRdQueue
	variable socketWrQueue
	variable socketPhQueue
	variable socketClosing
	variable socketPlayCmd
	variable socketCoEvent
	variable socketProxyId
	if {[info exists socketMapping]} {
	    # Close open sockets on re-init.  Do not permit retries.
	    foreach {url sock} [array get socketMapping] {
		unset -nocomplain socketClosing($url)
		unset -nocomplain socketPlayCmd($url)
		CloseSocket $sock
	    }
	}

	# CloseSocket should have unset the socket* arrays, one element at
	# a time.  Now unset anything that was overlooked.
	# Traces on "unset socketRdState(*)" will call CancelReadPipeline and
	# cancel any queued responses.
	# Traces on "unset socketWrState(*)" will call CancelWritePipeline and
	# cancel any queued requests.
	array unset socketMapping
	array unset socketRdState
	array unset socketWrState
	array unset socketRdQueue
	array unset socketWrQueue
	array unset socketPhQueue
	array unset socketClosing
	array unset socketPlayCmd
	array unset socketCoEvent
	array unset socketProxyId
	array set socketMapping {}
	array set socketRdState {}
	array set socketWrState {}
	array set socketRdQueue {}
	array set socketWrQueue {}
	array set socketPhQueue {}
	array set socketClosing {}
	array set socketPlayCmd {}
	array set socketCoEvent {}
	array set socketProxyId {}
	return
    }
    init

    variable urlTypes
    if {![info exists urlTypes]} {
	set urlTypes(http) [list 80 ::http::AltSocket {} 1 0]
    }

    variable encodings [string tolower [encoding names]]
    # This can be changed, but iso8859-1 is the RFC standard.
    variable defaultCharset
    if {![info exists defaultCharset]} {
	set defaultCharset "iso8859-1"
    }

    # Force RFC 3986 strictness in geturl url verification?
    variable strict
    if {![info exists strict]} {
	set strict 1
    }

    # Let user control default keepalive for compatibility
    variable defaultKeepalive
    if {![info exists defaultKeepalive]} {
	set defaultKeepalive 0
    }

    # Regular expression used to parse cookies
    variable CookieRE {(?x)                            # EXPANDED SYNTAX
	\s*                                            # Ignore leading spaces
	([^][\u0000- ()<>@,;:\\""/?={}\u007f-\uffff]+) # Match the name
	=                                              # LITERAL: Equal sign
	([!\u0023-+\u002D-:<-\u005B\u005D-~]*)         # Match the value
	(?:
	 \s* ; \s*                                     # LITERAL: semicolon
	 ([^\u0000]+)                                  # Match the options
	)?
    }

    variable TmpSockCounter 0
    variable ThreadCounter  0

    variable reasonDict [dict create {*}{
	100 Continue
	101 {Switching Protocols}
	102 Processing
	103 {Early Hints}
	200 OK
	201 Created
	202 Accepted
	203 {Non-Authoritative Information}
	204 {No Content}
	205 {Reset Content}
	206 {Partial Content}
	207 Multi-Status
	208 {Already Reported}
	226 {IM Used}
	300 {Multiple Choices}
	301 {Moved Permanently}
	302 Found
	303 {See Other}
	304 {Not Modified}
	305 {Use Proxy}
	306 (Unused)
	307 {Temporary Redirect}
	308 {Permanent Redirect}
	400 {Bad Request}
	401 Unauthorized
	402 {Payment Required}
	403 Forbidden
	404 {Not Found}
	405 {Method Not Allowed}
	406 {Not Acceptable}
	407 {Proxy Authentication Required}
	408 {Request Timeout}
	409 Conflict
	410 Gone
	411 {Length Required}
	412 {Precondition Failed}
	413 {Content Too Large}
	414 {URI Too Long}
	415 {Unsupported Media Type}
	416 {Range Not Satisfiable}
	417 {Expectation Failed}
	418 (Unused)
	421 {Misdirected Request}
	422 {Unprocessable Content}
	423 Locked
	424 {Failed Dependency}
	425 {Too Early}
	426 {Upgrade Required}
	428 {Precondition Required}
	429 {Too Many Requests}
	431 {Request Header Fields Too Large}
	451 {Unavailable For Legal Reasons}
	500 {Internal Server Error}
	501 {Not Implemented}
	502 {Bad Gateway}
	503 {Service Unavailable}
	504 {Gateway Timeout}
	505 {HTTP Version Not Supported}
	506 {Variant Also Negotiates}
	507 {Insufficient Storage}
	508 {Loop Detected}
	510 {Not Extended (OBSOLETED)}
	511 {Network Authentication Required}
    }]

    variable failedProxyValues {
	binary
	body
	charset
	coding
	connection
	connectionRespFlag
	currentsize
	host
	http
	httpResponse
	meta
	method
	querylength
	queryoffset
	reasonPhrase
	requestHeaders
	requestLine
	responseCode
	state
	status
	tid
	totalsize
	transfer
	type
    }

    namespace export geturl config reset wait formatQuery postError quoteString
    namespace export register unregister registerError
    namespace export requestLine requestHeaders requestHeaderValue
    namespace export responseLine responseHeaders responseHeaderValue
    namespace export responseCode responseBody responseInfo reasonPhrase
    # - Legacy aliases, were never exported:
    #     data, code, mapReply, meta, ncode
    # - Callable from outside (e.g. from TLS) by fully-qualified name, but
    #   not exported:
    #     socket
    # - Useful, but never exported (and likely to have naming collisions):
    #     size, status, cleanup, error, init
    #   Comments suggest that "init" can be used for re-initialisation,
    #   although the command is undocumented.
    # - Never exported, renamed from lower-case names:
    #   GetTextLine, MakeTransformationChunked.
}

# http::Log --
#
#	Debugging output -- define this to observe HTTP/1.1 socket usage.
#	Should echo any args received.
#
# Arguments:
#     msg	Message to output
#
if {[info command http::Log] eq {}} {proc http::Log {args} {}}

# http::register --
#
#     See documentation for details.
#
# Arguments:
#     proto		URL protocol prefix, e.g. https
#     port		Default port for protocol
#     command		Command to use to create socket
#     socketCmdVarName	(optional) name of variable provided by the protocol
#                       handler whose value is the callback used by argument
#                       "command" to open a socket. The default value "::socket"
#                       will be overwritten by http.
#     useSockThread	(optional, boolean)
#     endToEndProxy	(optional, boolean)
# Results:
#     list of port, command, variable name, (boolean) threadability,
#     and (boolean) endToEndProxy that was registered.

proc http::register {proto port command {socketCmdVarName {}} {useSockThread 0} {endToEndProxy 0}} {
    variable urlTypes
    set lower [string tolower $proto]
    if {[info exists urlTypes($lower)]} {
	unregister $lower
    }
    set urlTypes($lower) [list $port $command $socketCmdVarName $useSockThread $endToEndProxy]

    # If the external handler for protocol $proto has given $socketCmdVarName the expected
    # value "::socket", overwrite it with the new value.
    if {($socketCmdVarName ne {}) && ([set $socketCmdVarName] eq {::socket})} {
	set $socketCmdVarName ::http::socketAsCallback
    }

    return $urlTypes($lower)
}

# http::unregister --
#
#     Unregisters URL protocol handler
#
# Arguments:
#     proto	URL protocol prefix, e.g. https
# Results:
#     list of port, command, variable name, (boolean) useSockThread,
#     and (boolean) endToEndProxy that was unregistered.

proc http::unregister {proto} {
    variable urlTypes
    set lower [string tolower $proto]
    if {![info exists urlTypes($lower)]} {
	return -code error "unsupported url type \"$proto\""
    }
    set old $urlTypes($lower)

    # Restore the external handler's original value for $socketCmdVarName.
    lassign $old defport defcmd socketCmdVarName useSockThread endToEndProxy
    if {($socketCmdVarName ne {}) && ([set $socketCmdVarName] eq {::http::socketAsCallback})} {
	set $socketCmdVarName ::socket
    }

    unset urlTypes($lower)
    return $old
}

# http::config --
#
#	See documentation for details.
#
# Arguments:
#	args		Options parsed by the procedure.
# Results:
#	TODO

proc http::config {args} {
    variable http
    set options [lsort [array names http -*]]
    set usage [join $options ", "]
    if {[llength $args] == 0} {
	set result {}
	foreach name $options {
	    lappend result $name $http($name)
	}
	return $result
    }
    set options [string map {- ""} $options]
    set pat ^-(?:[join $options |])$
    if {[llength $args] == 1} {
	set flag [lindex $args 0]
	if {![regexp -- $pat $flag]} {
	    return -code error "Unknown option $flag, must be: $usage"
	}
	return $http($flag)
    } elseif {[llength $args] % 2} {
	return -code error "If more than one argument is supplied, the\
		number of arguments must be even"
    } else {
	foreach {flag value} $args {
	    if {![regexp -- $pat $flag]} {
		return -code error "Unknown option $flag, must be: $usage"
	    }
	    if {($flag eq {-threadlevel}) && ($value ni {0 1 2})} {
		return -code error {Option -threadlevel must be 0, 1 or 2}
	    }
	    set http($flag) $value
	}
	return
    }
}

# ------------------------------------------------------------------------------
#  Proc http::reasonPhrase
# ------------------------------------------------------------------------------
# Command to return the IANA-recommended "reason phrase" for a HTTP Status Code.
# Information obtained from:
# https://www.iana.org/assignments/http-status-codes/http-status-codes.xhtml
#
# Arguments:
# code        - A valid HTTP Status Code (integer from 100 to 599)
#
# Return Value: the reason phrase
# ------------------------------------------------------------------------------

proc http::reasonPhrase {code} {
    variable reasonDict
    if {![regexp -- {^[1-5][0-9][0-9]$} $code]} {
	set msg {argument must be a three-digit integer from 100 to 599}
	return -code error $msg
    }
    if {[dict exists $reasonDict $code]} {
	set reason [dict get $reasonDict $code]
    } else {
	set reason Unassigned
    }
    return $reason
}

# http::Finish --
#
#	Clean up the socket and eval close time callbacks
#
# Arguments:
#	token	    Connection token.
#	errormsg    (optional) If set, forces status to error.
#	skipCB      (optional) If set, don't call the -command callback. This
#		    is useful when geturl wants to throw an exception instead
#		    of calling the callback. That way, the same error isn't
#		    reported to two places.
#
# Side Effects:
#	May close the socket.

proc http::Finish {token {errormsg ""} {skipCB 0}} {
    variable socketMapping
    variable socketRdState
    variable socketWrState
    variable socketRdQueue
    variable socketWrQueue
    variable socketPhQueue
    variable socketClosing
    variable socketPlayCmd
    variable socketCoEvent
    variable socketProxyId

    variable $token
    upvar 0 $token state
    global errorInfo errorCode
    set closeQueue 0
    if {$errormsg ne ""} {
	set state(error) [list $errormsg $errorInfo $errorCode]
	set state(status) "error"
    }
    if {[info commands ${token}--EventCoroutine] ne {}} {
	rename ${token}--EventCoroutine {}
    }
    if {[info commands ${token}--SocketCoroutine] ne {}} {
	rename ${token}--SocketCoroutine {}
    }
    if {[info exists state(socketcoro)]} {
	Log $token Cancel socket after-idle event (Finish)
	after cancel $state(socketcoro)
	unset state(socketcoro)
    }

    # Is this an upgrade request/response?
    set upgradeResponse \
	[expr {    [info exists state(upgradeRequest)]
		&& $state(upgradeRequest)
		&& [info exists state(http)]
		&& ([ncode $token] eq {101})
		&& [info exists state(connection)]
		&& ("upgrade" in $state(connection))
		&& [info exists state(upgrade)]
		&& ("" ne $state(upgrade))
	}]

    if {  ($state(status) eq "timeout")
       || ($state(status) eq "error")
       || ($state(status) eq "eof")
    } {
	set closeQueue 1
	set connId $state(socketinfo)
	if {[info exists state(sock)]} {
	    set sock $state(sock)
	    CloseSocket $state(sock) $token
	} else {
	    # When opening the socket and calling http::reset
	    # immediately, the socket may not yet exist.
	    # Test http-4.11 may come here.
	}
	if {$state(tid) ne {}} {
	    # When opening the socket in a thread, and calling http::reset
	    # immediately, the thread may still exist.
	    # Test http-4.11 may come here.
	    thread::release $state(tid)
	    set state(tid) {}
	}
    } elseif {$upgradeResponse} {
	# Special handling for an upgrade request/response.
	# - geturl ensures that this is not a "persistent" socket used for
	#   multiple HTTP requests, so a call to KeepSocket is not needed.
	# - Leave socket open, so a call to CloseSocket is not needed either.
	# - Remove fileevent bindings.  The caller will set its own bindings.
	# - THE CALLER MUST PROCESS THE UPGRADED SOCKET IN THE CALLBACK COMMAND
	#   PASSED TO http::geturl AS -command callback.
	catch {fileevent $state(sock) readable {}}
	catch {fileevent $state(sock) writable {}}
    } elseif {([info exists state(-keepalive)] && !$state(-keepalive))
	    || ([info exists state(connection)] && ("close" in $state(connection)))
    } {
	set closeQueue 1
	set connId $state(socketinfo)
	if {[info exists state(sock)]} {
	    set sock $state(sock)
	    CloseSocket $state(sock) $token
	} else {
	    # When opening the socket and calling http::reset
	    # immediately, the socket may not yet exist.
	    # Test http-4.11 may come here.
	}
    } elseif {
	  ([info exists state(-keepalive)] && $state(-keepalive))
       && ([info exists state(connection)] && ("close" ni $state(connection)))
    } {
	KeepSocket $token
    }
    if {[info exists state(after)]} {
	after cancel $state(after)
	unset state(after)
    }
    if {[info exists state(-command)] && (!$skipCB)
	    && (![info exists state(done-command-cb)])} {
	set state(done-command-cb) yes
	if {    [catch {namespace eval :: $state(-command) $token} err]
	     && ($errormsg eq "")
	} {
	    set state(error) [list $err $errorInfo $errorCode]
	    set state(status) error
	}
    }

    if {    $closeQueue
	 && [info exists socketMapping($connId)]
	 && ($socketMapping($connId) eq $sock)
    } {
	http::CloseQueuedQueries $connId $token
	# This calls Unset.  Other cases do not need the call.
    }
    return
}

# http::KeepSocket -
#
#	Keep a socket in the persistent sockets table and connect it to its next
#	queued task if possible.  Otherwise leave it idle and ready for its next
#	use.
#
#	If $socketClosing(*), then ("close" in $state(connection)) and therefore
#	this command will not be called by Finish.
#
# Arguments:
#	token	    Connection token.

proc http::KeepSocket {token} {
    variable http
    variable socketMapping
    variable socketRdState
    variable socketWrState
    variable socketRdQueue
    variable socketWrQueue
    variable socketPhQueue
    variable socketClosing
    variable socketPlayCmd
    variable socketCoEvent
    variable socketProxyId

    variable $token
    upvar 0 $token state
    set tk [namespace tail $token]

    # Keep this socket open for another request ("Keep-Alive").
    # React if the server half-closes the socket.
    # Discussion is in http::geturl.
    catch {fileevent $state(sock) readable [list http::CheckEof $state(sock)]}

    # The line below should not be changed in production code.
    # It is edited by the test suite.
    set TEST_EOF 0
    if {$TEST_EOF} {
	# ONLY for testing reaction to server eof.
	# No server timeouts will be caught.
	catch {fileevent $state(sock) readable {}}
    }

    if {    [info exists state(socketinfo)]
	 && [info exists socketMapping($state(socketinfo))]
    } {
	set connId $state(socketinfo)
	# The value "Rready" is set only here.
	set socketRdState($connId) Rready

	if {    $state(-pipeline)
	     && [info exists socketRdQueue($connId)]
	     && [llength $socketRdQueue($connId)]
	} {
	    # The usual case for pipelined responses - if another response is
	    # queued, arrange to read it.
	    set token3 [lindex $socketRdQueue($connId) 0]
	    set socketRdQueue($connId) [lrange $socketRdQueue($connId) 1 end]

	    #Log pipelined, GRANT read access to $token3 in KeepSocket
	    set socketRdState($connId) $token3
	    ReceiveResponse $token3

	    # Other pipelined cases.
	    # - The test above ensures that, for the pipelined cases in the two
	    #   tests below, the read queue is empty.
	    # - In those two tests, check whether the next write will be
	    #   nonpipeline.
	} elseif {
		$state(-pipeline)
	     && [info exists socketWrState($connId)]
	     && ($socketWrState($connId) eq "peNding")

	     && [info exists socketWrQueue($connId)]
	     && [llength $socketWrQueue($connId)]
	     && (![set token3 [lindex $socketWrQueue($connId) 0]
		   set ${token3}(-pipeline)
		  ]
		)
	} {
	    # This case:
	    # - Now it the time to run the "pending" request.
	    # - The next token in the write queue is nonpipeline, and
	    #   socketWrState has been marked "pending" (in
	    #   http::NextPipelinedWrite or http::geturl) so a new pipelined
	    #   request cannot jump the queue.
	    #
	    # Tests:
	    # - In this case the read queue (tested above) is empty and this
	    #   "pending" write token is in front of the rest of the write
	    #   queue.
	    # - The write state is not Wready and therefore appears to be busy,
	    #   but because it is "pending" we know that it is reserved for the
	    #   first item in the write queue, a non-pipelined request that is
	    #   waiting for the read queue to empty.  That has now happened: so
	    #   give that request read and write access.
	    set conn [set ${token3}(connArgs)]
	    #Log nonpipeline, GRANT r/w access to $token3 in KeepSocket
	    set socketRdState($connId) $token3
	    set socketWrState($connId) $token3
	    set socketWrQueue($connId) [lrange $socketWrQueue($connId) 1 end]
	    # Connect does its own fconfigure.
	    fileevent $state(sock) writable [list http::Connect $token3 {*}$conn]
	    #Log ---- $state(sock) << conn to $token3 for HTTP request (c)

	} elseif {
		$state(-pipeline)
	     && [info exists socketWrState($connId)]
	     && ($socketWrState($connId) eq "peNding")

	} {
	    # Should not come here.  The second block in the previous "elseif"
	    # test should be tautologous (but was needed in an earlier
	    # implementation) and will be removed after testing.
	    # If we get here, the value "pending" was assigned in error.
	    # This error would block the queue for ever.
	    Log ^X$tk <<<<< Error in queueing of requests >>>>> - token $token

	} elseif {
		$state(-pipeline)
	     && [info exists socketWrState($connId)]
	     && ($socketWrState($connId) eq "Wready")

	     && [info exists socketWrQueue($connId)]
	     && [llength $socketWrQueue($connId)]
	     && (![set token3 [lindex $socketWrQueue($connId) 0]
		   set ${token3}(-pipeline)
		  ]
		)
	} {
	    # This case:
	    # - The next token in the write queue is nonpipeline, and
	    #   socketWrState is Wready.  Get the next event from socketWrQueue.
	    # Tests:
	    # - In this case the read state (tested above) is Rready and the
	    #   write state (tested here) is Wready - there is no "pending"
	    #   request.
	    # Code:
	    # - The code is the same as the code below for the nonpipelined
	    #   case with a queued request.
	    set conn [set ${token3}(connArgs)]
	    #Log nonpipeline, GRANT r/w access to $token3 in KeepSocket
	    set socketRdState($connId) $token3
	    set socketWrState($connId) $token3
	    set socketWrQueue($connId) [lrange $socketWrQueue($connId) 1 end]
	    # Connect does its own fconfigure.
	    fileevent $state(sock) writable [list http::Connect $token3 {*}$conn]
	    #Log ---- $state(sock) << conn to $token3 for HTTP request (c)

	} elseif {
		(!$state(-pipeline))
	     && [info exists socketWrQueue($connId)]
	     && [llength $socketWrQueue($connId)]
	     && ("close" ni $state(connection))
	} {
	    # If not pipelined, (socketRdState eq Rready) tells us that we are
	    # ready for the next write - there is no need to check
	    # socketWrState. Write the next request, if one is waiting.
	    # If the next request is pipelined, it receives premature read
	    # access to the socket. This is not a problem.
	    set token3 [lindex $socketWrQueue($connId) 0]
	    set conn [set ${token3}(connArgs)]
	    #Log nonpipeline, GRANT r/w access to $token3 in KeepSocket
	    set socketRdState($connId) $token3
	    set socketWrState($connId) $token3
	    set socketWrQueue($connId) [lrange $socketWrQueue($connId) 1 end]
	    # Connect does its own fconfigure.
	    fileevent $state(sock) writable [list http::Connect $token3 {*}$conn]
	    #Log ---- $state(sock) << conn to $token3 for HTTP request (d)

	} elseif {(!$state(-pipeline))} {
	    set socketWrState($connId) Wready
	    # Rready and Wready and idle: nothing to do.
	}

    } else {
	CloseSocket $state(sock) $token
	# There is no socketMapping($state(socketinfo)), so it does not matter
	# that CloseQueuedQueries is not called.
    }
    return
}

# http::CheckEof -
#
#	Read from a socket and close it if eof.
#	The command is bound to "fileevent readable" on an idle socket, and
#	"eof" is the only event that should trigger the binding, occurring when
#	the server times out and half-closes the socket.
#
#	A read is necessary so that [eof] gives a meaningful result.
#	Any bytes sent are junk (or a bug).

proc http::CheckEof {sock} {
    set junk [read $sock]
    set n [string length $junk]
    if {$n} {
	Log "WARNING: $n bytes received but no HTTP request sent"
    }

    if {[catch {eof $sock} res] || $res} {
	# The server has half-closed the socket.
	# If a new write has started, its transaction will fail and
	# will then be error-handled.
	CloseSocket $sock
    }
    return
}

# http::CloseSocket -
#
#	Close a socket and remove it from the persistent sockets table.  If
#	possible an http token is included here but when we are called from a
#	fileevent on remote closure we need to find the correct entry - hence
#	the "else" block of the first "if" command.

proc http::CloseSocket {s {token {}}} {
    variable socketMapping
    variable socketRdState
    variable socketWrState
    variable socketRdQueue
    variable socketWrQueue
    variable socketPhQueue
    variable socketClosing
    variable socketPlayCmd
    variable socketCoEvent
    variable socketProxyId

    set tk [namespace tail $token]

    catch {fileevent $s readable {}}
    set connId {}
    if {$token ne ""} {
	variable $token
	upvar 0 $token state
	if {[info exists state(socketinfo)]} {
	    set connId $state(socketinfo)
	}
    } else {
	set map [array get socketMapping]
	set ndx [lsearch -exact $map $s]
	if {$ndx >= 0} {
	    incr ndx -1
	    set connId [lindex $map $ndx]
	}
    }
    if {    ($connId ne {})
	 && [info exists socketMapping($connId)]
	 && ($socketMapping($connId) eq $s)
    } {
	Log "Closing connection $connId (sock $socketMapping($connId))"
	if {[catch {close $socketMapping($connId)} err]} {
	    Log "Error closing connection: $err"
	}
	if {$token eq {}} {
	    # Cases with a non-empty token are handled by Finish, so the tokens
	    # are finished in connection order.
	    http::CloseQueuedQueries $connId
	}
    } else {
	Log "Closing socket $s (no connection info)"
	if {[catch {close $s} err]} {
	    Log "Error closing socket: $err"
	}
    }
    return
}

# http::CloseQueuedQueries
#
#	connId  - identifier "domain:port" for the connection
#	token   - (optional) used only for logging
#
# Called from http::CloseSocket and http::Finish, after a connection is closed,
# to clear the read and write queues if this has not already been done.

proc http::CloseQueuedQueries {connId {token {}}} {
    variable socketMapping
    variable socketRdState
    variable socketWrState
    variable socketRdQueue
    variable socketWrQueue
    variable socketPhQueue
    variable socketClosing
    variable socketPlayCmd
    variable socketCoEvent
    variable socketProxyId

    ##Log CloseQueuedQueries $connId $token
    if {![info exists socketMapping($connId)]} {
	# Command has already been called.
	# Don't come here again - especially recursively.
	return
    }

    # Used only for logging.
    if {$token eq {}} {
	set tk {}
    } else {
	set tk [namespace tail $token]
    }

    if {    [info exists socketPlayCmd($connId)]
	 && ($socketPlayCmd($connId) ne {ReplayIfClose Wready {} {}})
    } {
	# Before unsetting, there is some unfinished business.
	# - If the server sent "Connection: close", we have stored the command
	#   for retrying any queued requests in socketPlayCmd, so copy that
	#   value for execution below.  socketClosing(*) was also set.
	# - Also clear the queues to prevent calls to Finish that would set the
	#   state for the requests that will be retried to "finished with error
	#   status".
	# - At this stage socketPhQueue is empty.
	set unfinished $socketPlayCmd($connId)
	set socketRdQueue($connId) {}
	set socketWrQueue($connId) {}
    } else {
	set unfinished {}
    }

    Unset $connId

    if {$unfinished ne {}} {
	Log ^R$tk Any unfinished transactions (excluding $token) failed \
		- token $token - unfinished $unfinished
	{*}$unfinished
	# Calls ReplayIfClose.
    }
    return
}

# http::Unset
#
#	The trace on "unset socketRdState(*)" will call CancelReadPipeline
#	and cancel any queued responses.
#	The trace on "unset socketWrState(*)" will call CancelWritePipeline
#	and cancel any queued requests.

proc http::Unset {connId} {
    variable socketMapping
    variable socketRdState
    variable socketWrState
    variable socketRdQueue
    variable socketWrQueue
    variable socketPhQueue
    variable socketClosing
    variable socketPlayCmd
    variable socketCoEvent
    variable socketProxyId

    unset socketMapping($connId)
    unset socketRdState($connId)
    unset socketWrState($connId)
    unset -nocomplain socketRdQueue($connId)
    unset -nocomplain socketWrQueue($connId)
    unset -nocomplain socketClosing($connId)
    unset -nocomplain socketPlayCmd($connId)
    unset -nocomplain socketProxyId($connId)
    return
}

# http::reset --
#
#	See documentation for details.
#
# Arguments:
#	token	Connection token.
#	why	Status info.
#
# Side Effects:
#	See Finish

proc http::reset {token {why reset}} {
    variable $token
    upvar 0 $token state
    set state(status) $why
    catch {fileevent $state(sock) readable {}}
    catch {fileevent $state(sock) writable {}}
    Finish $token
    if {[info exists state(error)]} {
	set errorlist $state(error)
	unset state
	eval ::error $errorlist
	# i.e. error msg errorInfo errorCode
    }
    return
}

# http::geturl --
#
#	Establishes a connection to a remote url via http.
#
# Arguments:
#	url		The http URL to goget.
#	args		Option value pairs. Valid options include:
#				-blocksize, -validate, -headers, -timeout
# Results:
#	Returns a token for this connection. This token is the name of an
#	array that the caller should unset to garbage collect the state.

proc http::geturl {url args} {
    variable urlTypes

    # - If ::tls::socketCmd has its default value "::socket", change it to the
    #   new value ::http::socketAsCallback.
    # - If the old value is different, then it has been modified either by the
    #   script or by the Tcl installation, and replaced by a new command.  The
    #   script or installation that modified ::tls::socketCmd is also
    #   responsible for integrating ::http::socketAsCallback into its own "new"
    #   command, if it wishes to do so.
    # - Commands that open a socket:
    #   - ::socket                 - basic
    #   - ::http::AltSocket        - can use a thread to avoid blockage by slow
    #                                DNS lookup.  See http::config option
    #                                -threadlevel.
    #   - ::http::socketAsCallback - as ::http::AltSocket, but can also open a
    #                                socket for HTTPS/TLS through a proxy.

    set token [CreateToken $url {*}$args]
    variable $token
    upvar 0 $token state

    AsyncTransaction $token

    # --------------------------------------------------------------------------
    # Synchronous Call to http::geturl
    # --------------------------------------------------------------------------
    # - If the call to http::geturl is asynchronous, it is now complete (apart
    #   from delivering the return value).
    # - If the call to http::geturl is synchronous, the command must now wait
    #   for the HTTP transaction to be completed.  The call to http::wait uses
    #   vwait, which may be inappropriate if the caller makes other HTTP
    #   requests in the background.
    # --------------------------------------------------------------------------

    if {![info exists state(-command)]} {
	# geturl does EVERYTHING asynchronously, so if the user
	# calls it synchronously, we just do a wait here.
	http::wait $token

	if {![info exists state]} {
	    # If we timed out then Finish has been called and the users
	    # command callback may have cleaned up the token. If so we end up
	    # here with nothing left to do.
	    return $token
	} elseif {$state(status) eq "error"} {
	    # Something went wrong while trying to establish the connection.
	    # Clean up after events and such, but DON'T call the command
	    # callback (if available) because we're going to throw an
	    # exception from here instead.
	    set err [lindex $state(error) 0]
	    cleanup $token
	    return -code error $err
	}
    }

    return $token
}

# ------------------------------------------------------------------------------
#  Proc http::CreateToken
# ------------------------------------------------------------------------------
# Command to convert arguments into an initialised request token.
# The return value is the variable name of the token.
#
# Other effects:
# - Sets ::http::http(usingThread) if not already done
# - Sets ::http::http(uid) if not already done
# - Increments ::http::http(uid)
# - May increment ::http::TmpSockCounter
# - Alters ::http::socketPlayCmd, ::http::socketWrQueue if a -keepalive 1
#   request is appended to the queue of a persistent socket that is already
#   scheduled to close.
#   This also sets state(alreadyQueued) to 1.
# - Alters ::http::socketPhQueue if a -keepalive 1 request is appended to the
#   queue of a persistent socket that has not yet been created (and is therefore
#   represented by a placeholder).
#   This also sets state(ReusingPlaceholder) to 1.
# ------------------------------------------------------------------------------

proc http::CreateToken {url args} {
    variable http
    variable urlTypes
    variable defaultCharset
    variable defaultKeepalive
    variable strict
    variable TmpSockCounter

    # Initialize the state variable, an array. We'll return the name of this
    # array as the token for the transaction.

    if {![info exists http(usingThread)]} {
	set http(usingThread) 0
    }
    if {![info exists http(uid)]} {
	set http(uid) 0
    }
    set token [namespace current]::[incr http(uid)]
    ##Log Starting http::geturl - token $token
    variable $token
    upvar 0 $token state
    set tk [namespace tail $token]
    reset $token
    Log ^A$tk URL $url - token $token

    # Process command options.

    array set state {
	-binary		false
	-blocksize	8192
	-queryblocksize 8192
	-validate	0
	-headers	{}
	-timeout	0
	-type		application/x-www-form-urlencoded
	-queryprogress	{}
	-protocol	1.1
	-guesstype      0
	binary		0
	state		created
	meta		{}
	method		{}
	coding		{}
	currentsize	0
	totalsize	0
	querylength	0
	queryoffset	0
	type		application/octet-stream
	body		{}
	status		""
	http		""
	httpResponse    {}
	responseCode    {}
	reasonPhrase    {}
	connection	keep-alive
	tid             {}
	requestHeaders  {}
	requestLine     {}
	transfer        {}
	proxyUsed       none
	protoSockThread 0
	protoProxyConn  0
    }
    set state(-keepalive) $defaultKeepalive
    set state(-strict) $strict
    # These flags have their types verified [Bug 811170]
    array set type {
	-binary		boolean
	-blocksize	integer
	-guesstype      boolean
	-queryblocksize integer
	-strict		boolean
	-timeout	integer
	-validate	boolean
	-headers	list
    }
    set state(charset)	$defaultCharset
    set options {
	-binary -blocksize -channel -command -guesstype -handler -headers -keepalive
	-method -myaddr -progress -protocol -query -queryblocksize
	-querychannel -queryprogress -strict -timeout -type -validate
    }
    set usage [join [lsort $options] ", "]
    set options [string map {- ""} $options]
    set pat ^-(?:[join $options |])$
    foreach {flag value} $args {
	if {[regexp -- $pat $flag]} {
	    # Validate numbers
	    if {    [info exists type($flag)]
		    && (![string is $type($flag) -strict $value])
	    } {
		unset $token
		return -code error \
		    "Bad value for $flag ($value), must be $type($flag)"
	    }
	    if {($flag eq "-headers") && ([llength $value] % 2 != 0)} {
		unset $token
		return -code error "Bad value for $flag ($value), number\
			of list elements must be even"
	    }
	    set state($flag) $value
	} else {
	    unset $token
	    return -code error "Unknown option $flag, can be: $usage"
	}
    }

    # Make sure -query and -querychannel aren't both specified

    set isQueryChannel [info exists state(-querychannel)]
    set isQuery [info exists state(-query)]
    if {$isQuery && $isQueryChannel} {
	unset $token
	return -code error "Can't combine -query and -querychannel options!"
    }

    # Validate URL, determine the server host and port, and check proxy case
    # Recognize user:pass@host URLs also, although we do not do anything with
    # that info yet.

    # URLs have basically four parts.
    # First, before the colon, is the protocol scheme (e.g. http)
    # Second, for HTTP-like protocols, is the authority
    #	The authority is preceded by // and lasts up to (but not including)
    #	the following / or ? and it identifies up to four parts, of which
    #	only one, the host, is required (if an authority is present at all).
    #	All other parts of the authority (user name, password, port number)
    #	are optional.
    # Third is the resource name, which is split into two parts at a ?
    #	The first part (from the single "/" up to "?") is the path, and the
    #	second part (from that "?" up to "#") is the query. *HOWEVER*, we do
    #	not need to separate them; we send the whole lot to the server.
    #	Both, path and query are allowed to be missing, including their
    #	delimiting character.
    # Fourth is the fragment identifier, which is everything after the first
    #	"#" in the URL. The fragment identifier MUST NOT be sent to the server
    #	and indeed, we don't bother to validate it (it could be an error to
    #	pass it in here, but it's cheap to strip).
    #
    # An example of a URL that has all the parts:
    #
    #     http://jschmoe:xyzzy@www.bogus.net:8000/foo/bar.tml?q=foo#changes
    #
    # The "http" is the protocol, the user is "jschmoe", the password is
    # "xyzzy", the host is "www.bogus.net", the port is "8000", the path is
    # "/foo/bar.tml", the query is "q=foo", and the fragment is "changes".
    #
    # Note that the RE actually combines the user and password parts, as
    # recommended in RFC 3986. Indeed, that RFC states that putting passwords
    # in URLs is a Really Bad Idea, something with which I would agree utterly.
    # RFC 9110 Sec 4.2.4 goes further than this, and deprecates the format
    # "user:password@".  It is retained here for backward compatibility,
    # but its use is not recommended.
    #
    # From a validation perspective, we need to ensure that the parts of the
    # URL that are going to the server are correctly encoded.  This is only
    # done if $state(-strict) is true (inherited from $::http::strict).

    set URLmatcher {(?x)		# this is _expanded_ syntax
	^
	(?: (\w+) : ) ?			# <protocol scheme>
	(?: //
	    (?:
		(
		    [^@/\#?]+		# <userinfo part of authority>
		) @
	    )?
	    (				# <host part of authority>
		[^/:\#?]+ |		# host name or IPv4 address
		\[ [^/\#?]+ \]		# IPv6 address in square brackets
	    )
	    (?: : (\d+) )?		# <port part of authority>
	)?
	( [/\?] [^\#]*)?		# <path> (including query)
	(?: \# (.*) )?			# <fragment>
	$
    }

    # Phase one: parse
    if {![regexp -- $URLmatcher $url -> proto user host port srvurl]} {
	unset $token
	return -code error "Unsupported URL: $url"
    }
    # Phase two: validate
    set host [string trim $host {[]}]; # strip square brackets from IPv6 address
    if {$host eq ""} {
	# Caller has to provide a host name; we do not have a "default host"
	# that would enable us to handle relative URLs.
	unset $token
	return -code error "Missing host part: $url"
	# Note that we don't check the hostname for validity here; if it's
	# invalid, we'll simply fail to resolve it later on.
    }
    if {$port ne "" && $port > 65535} {
	unset $token
	return -code error "Invalid port number: $port"
    }
    # The user identification and resource identification parts of the URL can
    # have encoded characters in them; take care!
    if {$user ne ""} {
	# Check for validity according to RFC 3986, Appendix A
	set validityRE {(?xi)
	    ^
	    (?: [-\w.~!$&'()*+,;=:] | %[0-9a-f][0-9a-f] )+
	    $
	}
	if {$state(-strict) && ![regexp -- $validityRE $user]} {
	    unset $token
	    # Provide a better error message in this error case
	    if {[regexp {(?i)%(?![0-9a-f][0-9a-f]).?.?} $user bad]} {
		return -code error \
			"Illegal encoding character usage \"$bad\" in URL user"
	    }
	    return -code error "Illegal characters in URL user"
	}
    }
    if {$srvurl ne ""} {
	# RFC 3986 allows empty paths (not even a /), but servers
	# return 400 if the path in the HTTP request doesn't start
	# with / , so add it here if needed.
	if {[string index $srvurl 0] ne "/"} {
	    set srvurl /$srvurl
	}
	# Check for validity according to RFC 3986, Appendix A
	set validityRE {(?xi)
	    ^
	    # Path part (already must start with / character)
	    (?:	      [-\w.~!$&'()*+,;=:@/]  | %[0-9a-f][0-9a-f] )*
	    # Query part (optional, permits ? characters)
	    (?: \? (?: [-\w.~!$&'()*+,;=:@/?] | %[0-9a-f][0-9a-f] )* )?
	    $
	}
	if {$state(-strict) && ![regexp -- $validityRE $srvurl]} {
	    unset $token
	    # Provide a better error message in this error case
	    if {[regexp {(?i)%(?![0-9a-f][0-9a-f])..} $srvurl bad]} {
		return -code error \
		    "Illegal encoding character usage \"$bad\" in URL path"
	    }
	    return -code error "Illegal characters in URL path"
	}
	if {![regexp {^[^?#]+} $srvurl state(path)]} {
	    set state(path) /
	}
    } else {
	set srvurl /
	set state(path) /
    }
    if {$proto eq ""} {
	set proto http
    }
    set lower [string tolower $proto]
    if {![info exists urlTypes($lower)]} {
	unset $token
	return -code error "Unsupported URL type \"$proto\""
    }
    lassign $urlTypes($lower) defport defcmd socketCmdVarName useSockThread end2EndProxy

    # If the external handler for protocol $proto has given $socketCmdVarName the expected
    # value "::socket", overwrite it with the new value.
    if {($socketCmdVarName ne {}) && ([set $socketCmdVarName] eq {::socket})} {
	set $socketCmdVarName ::http::socketAsCallback
    }

    set state(protoSockThread) $useSockThread
    set state(protoProxyConn) $end2EndProxy

    if {$port eq ""} {
	set port $defport
    }
    if {![catch {$http(-proxyfilter) $host} proxy]} {
	set phost [lindex $proxy 0]
	set pport [lindex $proxy 1]
    } else {
	set phost {}
	set pport {}
    }

    # OK, now reassemble into a full URL
    set url ${proto}://
    if {$user ne ""} {
	append url $user
	append url @
    }
    append url $host
    if {$port != $defport} {
	append url : $port
    }
    append url $srvurl
    # Don't append the fragment! RFC 7230 Sec 5.1
    set state(url) $url

    # Proxy connections aren't shared among different hosts.
    set state(socketinfo) $host:$port

    # Save the accept types at this point to prevent a race condition. [Bug
    # c11a51c482]
    set state(accept-types) $http(-accept)

    # Check whether this is an Upgrade request.
    set connectionValues [SplitCommaSeparatedFieldValue \
			      [GetFieldValue $state(-headers) Connection]]
    set connectionValues [string tolower $connectionValues]
    set upgradeValues [SplitCommaSeparatedFieldValue \
			   [GetFieldValue $state(-headers) Upgrade]]
    set state(upgradeRequest) [expr {    "upgrade" in $connectionValues
				      && [llength $upgradeValues] >= 1}]
    set state(connectionValues) $connectionValues

    if {$isQuery || $isQueryChannel} {
	# It's a POST.
	# A client wishing to send a non-idempotent request SHOULD wait to send
	# that request until it has received the response status for the
	# previous request.
	if {$http(-postfresh)} {
	    # Override -keepalive for a POST.  Use a new connection, and thus
	    # avoid the small risk of a race against server timeout.
	    set state(-keepalive) 0
	} else {
	    # Allow -keepalive but do not -pipeline - wait for the previous
	    # transaction to finish.
	    # There is a small risk of a race against server timeout.
	    set state(-pipeline) 0
	}
    } elseif {$state(upgradeRequest)} {
	# It's an upgrade request.  Method must be GET (untested).
	# Force -keepalive to 0 so the connection is not made over a persistent
	# socket, i.e. one used for multiple HTTP requests.
	set state(-keepalive) 0
    } else {
	# It's a non-upgrade GET or HEAD.
	set state(-pipeline) $http(-pipeline)
    }

    # We cannot handle chunked encodings with -handler, so force HTTP/1.0
    # until we can manage this.
    if {[info exists state(-handler)]} {
	set state(-protocol) 1.0
    }

    # RFC 7320 A.1 - HTTP/1.0 Keep-Alive is problematic. We do not support it.
    if {$state(-protocol) eq "1.0"} {
	set state(connection) close
	set state(-keepalive) 0
    }

    # Handle proxy requests here for http:// but not for https://
    # The proxying for https is done in the ::http::socketAsCallback command.
    # A proxy request for http:// needs the full URL in the HTTP request line,
    # including the server name.
    # The *tls* test below attempts to describe protocols in addition to
    # "https on port 443" that use HTTP over TLS.
    if {($phost ne "") && (!$end2EndProxy)} {
	set srvurl $url
	set targetAddr [list $phost $pport]
	set state(proxyUsed) HttpProxy
	# The value of state(proxyUsed) none|HttpProxy depends only on the
	# all-transactions http::config settings and on the target URL.
	# Even if this is a persistent socket there is no need to change the
	# value of state(proxyUsed) for other transactions that use the socket:
	# they have the same value already.
    } else {
	set targetAddr [list $host $port]
    }

    set sockopts [list -async]

    # Pass -myaddr directly to the socket command
    if {[info exists state(-myaddr)]} {
	lappend sockopts -myaddr $state(-myaddr)
    }

    if {$useSockThread} {
	set targs [list -type $token]
    } else {
	set targs {}
    }
    set state(connArgs) [list $proto $phost $srvurl]
    set state(openCmd) [list {*}$defcmd {*}$sockopts {*}$targs {*}$targetAddr]

    # See if we are supposed to use a previously opened channel.
    # - In principle, ANY call to http::geturl could use a previously opened
    #   channel if it is available - the "Connection: keep-alive" header is a
    #   request to leave the channel open AFTER completion of this call.
    # - In fact, we try to use an existing channel only if -keepalive 1 -- this
    #   means that at most one channel is left open for each value of
    #   $state(socketinfo). This property simplifies the mapping of open
    #   channels.
    set reusing 0
    set state(alreadyQueued) 0
    set state(ReusingPlaceholder) 0
    if {$state(-keepalive)} {
	variable socketMapping
	variable socketRdState
	variable socketWrState
	variable socketRdQueue
	variable socketWrQueue
	variable socketPhQueue
	variable socketClosing
	variable socketPlayCmd
	variable socketCoEvent
	variable socketProxyId

	if {[info exists socketMapping($state(socketinfo))]} {
	    # - If the connection is idle, it has a "fileevent readable" binding
	    #   to http::CheckEof, in case the server times out and half-closes
	    #   the socket (http::CheckEof closes the other half).
	    # - We leave this binding in place until just before the last
	    #   puts+flush in http::Connected (GET/HEAD) or http::Write (POST),
	    #   after which the HTTP response might be generated.

	    if {    [info exists socketClosing($state(socketinfo))]
		       && $socketClosing($state(socketinfo))
	    } {
		# socketClosing(*) is set because the server has sent a
		# "Connection: close" header.
		# Do not use the persistent socket again.
		# Since we have only one persistent socket per server, and the
		# old socket is not yet dead, add the request to the write queue
		# of the dying socket, which will be replayed by ReplayIfClose.
		# Also add it to socketWrQueue(*) which is used only if an error
		# causes a call to Finish.
		set reusing 1
		set sock $socketMapping($state(socketinfo))
		set state(proxyUsed) $socketProxyId($state(socketinfo))
		Log "reusing closing socket $sock for $state(socketinfo) - token $token"

		set state(alreadyQueued) 1
		lassign $socketPlayCmd($state(socketinfo)) com0 com1 com2 com3
		lappend com3 $token
		set socketPlayCmd($state(socketinfo)) [list $com0 $com1 $com2 $com3]
		lappend socketWrQueue($state(socketinfo)) $token
		##Log socketPlayCmd($state(socketinfo)) is $socketPlayCmd($state(socketinfo))
		##Log socketWrQueue($state(socketinfo)) is $socketWrQueue($state(socketinfo))
	    } elseif {
		   [catch {fconfigure $socketMapping($state(socketinfo))}]
		&& (![SockIsPlaceHolder $socketMapping($state(socketinfo))])
	    } {
		###Log "Socket $socketMapping($state(socketinfo)) for $state(socketinfo)"
		# FIXME Is it still possible for this code to be executed? If
		#       so, this could be another place to call TestForReplay,
		#       rather than discarding the queued transactions.
		Log "WARNING: socket for $state(socketinfo) was closed\
			- token $token"
		Log "WARNING - if testing, pay special attention to this\
			case (GH) which is seldom executed - token $token"

		# This will call CancelReadPipeline, CancelWritePipeline, and
		# cancel any queued requests, responses.
		Unset $state(socketinfo)
	    } else {
		# Use the persistent socket.
		# - The socket may not be ready to write: an earlier request might
		#   still be still writing (in the pipelined case) or
		#   writing/reading (in the nonpipeline case). This possibility
		#   is handled by socketWrQueue later in this command.
		# - The socket may not yet exist, and be defined with a placeholder.
		set reusing 1
		set sock $socketMapping($state(socketinfo))
		set state(proxyUsed) $socketProxyId($state(socketinfo))
		if {[SockIsPlaceHolder $sock]} {
		    set state(ReusingPlaceholder) 1
		    lappend socketPhQueue($sock) $token
		}
		Log "reusing open socket $sock for $state(socketinfo) - token $token"
	    }
	    # Do not automatically close the connection socket.
	    set state(connection) keep-alive
	}
    }

    set state(reusing) $reusing
    unset reusing

    if {![info exists sock]} {
	# N.B. At this point ([info exists sock] == $state(reusing)).
	# This will no longer be true after we set a value of sock here.
	# Give the socket a placeholder name.
	set sock HTTP_PLACEHOLDER_[incr TmpSockCounter]
    }
    set state(sock) $sock

    if {$state(reusing)} {
	# Define these for use (only) by http::ReplayIfDead if the persistent
	# connection has died.
	set state(tmpConnArgs) $state(connArgs)
	set state(tmpState) [array get state]
	set state(tmpOpenCmd) $state(openCmd)
    }
    return $token
}


# ------------------------------------------------------------------------------
#  Proc ::http::SockIsPlaceHolder
# ------------------------------------------------------------------------------
# Command to return 0 if the argument is a genuine socket handle, or 1 if is a
# placeholder value generated by geturl or ReplayCore before the real socket is
# created.
#
# Arguments:
# sock        - either a valid socket handle or a placeholder value
#
# Return Value: 0 or 1
# ------------------------------------------------------------------------------

proc http::SockIsPlaceHolder {sock} {
    expr {[string range $sock 0 16] eq {HTTP_PLACEHOLDER_}}
}


# ------------------------------------------------------------------------------
# state(reusing)
# ------------------------------------------------------------------------------
# - state(reusing) is set by geturl, ReplayCore
# - state(reusing) is used by geturl, AsyncTransaction, OpenSocket,
#   ConfigureNewSocket, and ScheduleRequest when creating and configuring the
#   connection.
# - state(reusing) is used by Connect, Connected, Event x 2 when deciding
#   whether to call TestForReplay.
# - Other places where state(reusing) is used:
#   - Connected   - if reusing and not pipelined, start the state(-timeout)
#                   timeout (when writing).
#   - DoneRequest - if reusing and pipelined, send the next pipelined write
#   - Event       - if reusing and pipelined, start the state(-timeout)
#                   timeout (when reading).
#   - Event       - if (not reusing) and pipelined, send the next pipelined
#                   write.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#  Proc http::AsyncTransaction
# ------------------------------------------------------------------------------
# This command is called by geturl and ReplayCore to prepare the HTTP
# transaction prescribed by a suitably prepared token.
#
# Arguments:
# token         - connection token (name of an array)
#
# Return Value: none
# ------------------------------------------------------------------------------

proc http::AsyncTransaction {token} {
    variable $token
    upvar 0 $token state
    set tk [namespace tail $token]

    variable socketMapping
    variable socketRdState
    variable socketWrState
    variable socketRdQueue
    variable socketWrQueue
    variable socketPhQueue
    variable socketClosing
    variable socketPlayCmd
    variable socketCoEvent
    variable socketProxyId

    set sock $state(sock)

    # See comments above re the start of this timeout in other cases.
    if {(!$state(reusing)) && ($state(-timeout) > 0)} {
	set state(after) [after $state(-timeout) \
		[list http::reset $token timeout]]
    }

    if {    $state(-keepalive)
	 && (![info exists socketMapping($state(socketinfo))])
    } {
	# This code is executed only for the first -keepalive request on a
	# socket.  It makes the socket persistent.
	##Log "  PreparePersistentConnection" $token -- $sock -- DO
	set DoLater [PreparePersistentConnection $token]
    } else {
	##Log "  PreparePersistentConnection" $token -- $sock -- SKIP
	set DoLater {-traceread 0 -tracewrite 0}
    }

    if {$state(ReusingPlaceholder)} {
	# - This request was added to the socketPhQueue of a persistent
	#   connection.
	# - But the connection has not yet been created and is a placeholder;
	# - And the placeholder was created by an earlier request.
	# - When that earlier request calls OpenSocket, its placeholder is
	#   replaced with a true socket, and it then executes the equivalent of
	#   OpenSocket for any subsequent requests that have
	#   $state(ReusingPlaceholder).
	Log >J$tk after idle coro NO - ReusingPlaceholder
    } elseif {$state(alreadyQueued)} {
	# - This request was added to the socketWrQueue and socketPlayCmd
	#   of a persistent connection that will close at the end of its current
	#   read operation.
	Log >J$tk after idle coro NO - alreadyQueued
    } else {
	Log >J$tk after idle coro YES
	set CoroName ${token}--SocketCoroutine
	set cancel [after idle [list coroutine $CoroName ::http::OpenSocket \
		$token $DoLater]]
	dict set socketCoEvent($state(socketinfo)) $token $cancel
	set state(socketcoro) $cancel
    }

    return
}


# ------------------------------------------------------------------------------
#  Proc http::PreparePersistentConnection
# ------------------------------------------------------------------------------
# This command is called by AsyncTransaction to initialise a "persistent
# connection" based upon a socket placeholder.  It is called the first time the
# socket is associated with a "-keepalive" request.
#
# Arguments:
# token         - connection token (name of an array)
#
# Return Value: - DoLater, a dictionary of boolean values listing unfinished
#                 tasks; to be passed to ConfigureNewSocket via OpenSocket.
# ------------------------------------------------------------------------------

proc http::PreparePersistentConnection {token} {
    variable $token
    upvar 0 $token state

    variable socketMapping
    variable socketRdState
    variable socketWrState
    variable socketRdQueue
    variable socketWrQueue
    variable socketPhQueue
    variable socketClosing
    variable socketPlayCmd
    variable socketCoEvent
    variable socketProxyId

    set DoLater {-traceread 0 -tracewrite 0}
    set socketMapping($state(socketinfo)) $state(sock)
    set socketProxyId($state(socketinfo)) $state(proxyUsed)
    # - The value of state(proxyUsed) was set in http::CreateToken to either
    #   "none" or "HttpProxy".
    # - $token is the first transaction to use this placeholder, so there are
    #   no other tokens whose (proxyUsed) must be modified.

    if {![info exists socketRdState($state(socketinfo))]} {
	set socketRdState($state(socketinfo)) {}
	# set varName ::http::socketRdState($state(socketinfo))
	# trace add variable $varName unset ::http::CancelReadPipeline
	dict set DoLater -traceread 1
    }
    if {![info exists socketWrState($state(socketinfo))]} {
	set socketWrState($state(socketinfo)) {}
	# set varName ::http::socketWrState($state(socketinfo))
	# trace add variable $varName unset ::http::CancelWritePipeline
	dict set DoLater -tracewrite 1
    }

    if {$state(-pipeline)} {
	#Log new, init for pipelined, GRANT write access to $token in geturl
	# Also grant premature read access to the socket. This is OK.
	set socketRdState($state(socketinfo)) $token
	set socketWrState($state(socketinfo)) $token
    } else {
	# socketWrState is not used by this non-pipelined transaction.
	# We cannot leave it as "Wready" because the next call to
	# http::geturl with a pipelined transaction would conclude that the
	# socket is available for writing.
	#Log new, init for nonpipeline, GRANT r/w access to $token in geturl
	set socketRdState($state(socketinfo)) $token
	set socketWrState($state(socketinfo)) $token
    }

    # Value of socketPhQueue() may have already been set by ReplayCore.
    if {![info exists socketPhQueue($state(sock))]} {
	set socketPhQueue($state(sock))   {}
    }
    set socketRdQueue($state(socketinfo)) {}
    set socketWrQueue($state(socketinfo)) {}
    set socketClosing($state(socketinfo)) 0
    set socketPlayCmd($state(socketinfo)) {ReplayIfClose Wready {} {}}
    set socketCoEvent($state(socketinfo)) {}
    set socketProxyId($state(socketinfo)) {}

    return $DoLater
}

# ------------------------------------------------------------------------------
#  Proc ::http::OpenSocket
# ------------------------------------------------------------------------------
# This command is called as a coroutine idletask to start the asynchronous HTTP
# transaction in most cases.  For the exceptions, see the calling code in
# command AsyncTransaction.
#
# Arguments:
# token       - connection token (name of an array)
# DoLater     - dictionary of boolean values listing unfinished tasks
#
# Return Value: none
# ------------------------------------------------------------------------------

proc http::OpenSocket {token DoLater} {
    variable $token
    upvar 0 $token state
    set tk [namespace tail $token]

    variable socketMapping
    variable socketRdState
    variable socketWrState
    variable socketRdQueue
    variable socketWrQueue
    variable socketPhQueue
    variable socketClosing
    variable socketPlayCmd
    variable socketCoEvent
    variable socketProxyId

    Log >K$tk Start OpenSocket coroutine

    if {![info exists state(-keepalive)]} {
	# The request has already been cancelled by the calling script.
	return
    }

    set sockOld $state(sock)

    dict unset socketCoEvent($state(socketinfo)) $token
    unset -nocomplain state(socketcoro)

    if {[catch {
	if {$state(reusing)} {
	    # If ($state(reusing)) is true, then we do not need to create a new
	    # socket, even if $sockOld is only a placeholder for a socket.
	    set sock $sockOld
	} else {
	    # set sock in the [catch] below.
	    set pre [clock milliseconds]
	    ##Log pre socket opened, - token $token
	    ##Log $state(openCmd) - token $token
	    set sock [namespace eval :: $state(openCmd)]
	    set state(sock) $sock
	    # Normal return from $state(openCmd) always returns a valid socket.
	    # A TLS proxy connection with 407 or other failure from the
	    # proxy server raises an error.

	    # Initialisation of a new socket.
	    ##Log post socket opened, - token $token
	    ##Log socket opened, now fconfigure - token $token
	    set delay [expr {[clock milliseconds] - $pre}]
	    if {$delay > 3000} {
		Log socket delay $delay - token $token
	    }
	    fconfigure $sock -translation {auto crlf} \
			     -buffersize $state(-blocksize)
	    if {[package vsatisfies [package provide Tcl] 9.0-]} {
		fconfigure $sock -profile replace
	    }
	    ##Log socket opened, DONE fconfigure - token $token
	}

	Log "Using $sock for $state(socketinfo) - token $token" \
	    [expr {$state(-keepalive)?"keepalive":""}]

	# Code above has set state(sock) $sock
	ConfigureNewSocket $token $sockOld $DoLater
	##Log OpenSocket success $sock - token $token
    } result errdict]} {
	##Log OpenSocket failed $result - token $token
	# There may be other requests in the socketPhQueue.
	# Prepare socketPlayCmd so that Finish will replay them.
	if {    ($state(-keepalive)) && (!$state(reusing))
	     && [info exists socketPhQueue($sockOld)]
	     && ($socketPhQueue($sockOld) ne {})
	} {
	    if {$socketMapping($state(socketinfo)) ne $sockOld} {
		Log "WARNING: this code should not be reached.\
			{$socketMapping($state(socketinfo)) ne $sockOld}"
	    }
	    set socketPlayCmd($state(socketinfo)) [list ReplayIfClose Wready {} $socketPhQueue($sockOld)]
	    set socketPhQueue($sockOld) {}
	}
	if {[string range $result 0 20] eq {proxy connect failed:}} {
	    # - The HTTPS proxy did not create a socket.  The pre-existing value
	    #   (a "placeholder socket") is unchanged.
	    # - The proxy returned a valid HTTP response to the failed CONNECT
	    #   request, and http::SecureProxyConnect copied this to $token,
	    #   and also set ${token}(connection) set to "close".
	    # - Remove the error message $result so that Finish delivers this
	    #   HTTP response to the caller.
	    set result {}
	}
	Finish $token $result
	# Because socket creation failed, the placeholder "socket" must be
	# "closed" and (if persistent) removed from the persistent sockets
	# table.  In the {proxy connect failed:} case Finish does this because
	# the value of ${token}(connection) is "close". In the other cases here,
	# it does so because $result is non-empty.
    }
    ##Log Leaving http::OpenSocket coroutine [info coroutine] - token $token
    return
}


# ------------------------------------------------------------------------------
#  Proc ::http::ConfigureNewSocket
# ------------------------------------------------------------------------------
# Command to initialise a newly-created socket.  Called only from OpenSocket.
#
# This command is called by OpenSocket whenever a genuine socket (sockNew) has
# been opened for for use by HTTP.  It does two things:
# (1) If $token uses a placeholder socket, this command replaces the placeholder
#     socket with the real socket, not only in $token but in all other requests
#     that use the same placeholder.
# (2) It calls ScheduleRequest to schedule each request that uses the socket.
#
#
# Value of sockOld/sockNew can be "sock" (genuine socket) or "ph" (placeholder).
# sockNew is ${token}(sock)
# sockOld   sockNew  CASES
#  sock       sock   (if $reusing, and sockOld is sock)
#  ph         sock   (if (not $reusing), and sockOld is ph)
#  ph         ph     (if $reusing, and sockOld is ph) - not called in this case
#  sock       ph     (cannot occur unless a bug)      - not called in this case
#                    (if (not $reusing), and sockOld is sock) - illogical
#
# Arguments:
# token         - connection token (name of an array)
# sockOld       - handle or placeholder used for a socket before the call to
#                 OpenSocket
# DoLater       - dictionary of boolean values listing unfinished tasks
#
# Return Value: none
# ------------------------------------------------------------------------------

proc http::ConfigureNewSocket {token sockOld DoLater} {
    variable $token
    upvar 0 $token state
    set tk [namespace tail $token]

    variable socketMapping
    variable socketRdState
    variable socketWrState
    variable socketRdQueue
    variable socketWrQueue
    variable socketPhQueue
    variable socketClosing
    variable socketPlayCmd
    variable socketCoEvent
    variable socketProxyId

    set reusing $state(reusing)
    set sock $state(sock)
    set proxyUsed $state(proxyUsed)
    ##Log "  ConfigureNewSocket" $token $sockOld ... -- $reusing $sock $proxyUsed

    if {(!$reusing) && ($sock ne $sockOld)} {
	# Replace the placeholder value sockOld with sock.

	if {    [info exists socketMapping($state(socketinfo))]
	     && ($socketMapping($state(socketinfo)) eq $sockOld)
	} {
	    set socketMapping($state(socketinfo)) $sock
	    set socketProxyId($state(socketinfo)) $proxyUsed
	    # tokens that use the placeholder $sockOld are updated below.
	    ##Log set socketMapping($state(socketinfo)) $sock
	}

	# Now finish any tasks left over from PreparePersistentConnection on
	# the connection.
	#
	# The "unset" traces are fired by init (clears entire arrays), and
	# by http::Unset.
	# Unset is called by CloseQueuedQueries and (possibly never) by geturl.
	#
	# CancelReadPipeline, CancelWritePipeline call http::Finish for each
	# token.
	#
	# FIXME If Finish is placeholder-aware, these traces can be set earlier,
	# in PreparePersistentConnection.

	if {[dict get $DoLater -traceread]} {
	    set varName ::http::socketRdState($state(socketinfo))
	    trace add variable $varName unset ::http::CancelReadPipeline
	}
	if {[dict get $DoLater -tracewrite]} {
	    set varName ::http::socketWrState($state(socketinfo))
	    trace add variable $varName unset ::http::CancelWritePipeline
	}
    }

    # Do this in all cases.
    ScheduleRequest $token

    # Now look at all other tokens that use the placeholder $sockOld.
    if {    (!$reusing)
	 && ($sock ne $sockOld)
	 && [info exists socketPhQueue($sockOld)]
    } {
	##Log "  ConfigureNewSocket" $token scheduled, now do $socketPhQueue($sockOld)
	foreach tok $socketPhQueue($sockOld) {
	    # 1. Amend the token's (sock).
	    ##Log set ${tok}(sock) $sock
	    set ${tok}(sock) $sock
	    set ${tok}(proxyUsed) $proxyUsed

	    # 2. Schedule the token's HTTP request.
	    # Every token in socketPhQueue(*) has reusing 1 alreadyQueued 0.
	    set ${tok}(reusing) 1
	    set ${tok}(alreadyQueued) 0
	    ScheduleRequest $tok
	}
	set socketPhQueue($sockOld) {}
    }
    ##Log "  ConfigureNewSocket" $token DONE

    return
}


# ------------------------------------------------------------------------------
# The values of array variables socketMapping etc.
# ------------------------------------------------------------------------------
# connId                 "$host:$port"
# socketMapping($connId) the handle or placeholder for the socket that is used
#                        for "-keepalive 1" requests to $connId.
# socketRdState($connId) the token that is currently reading from the socket.
#                        Other values: Rready (ready for next token to read).
# socketWrState($connId) the token that is currently writing to the socket.
#                        Other values: Wready (ready for next token to write),
#                        peNding (would be ready for next write, except that
#                        the integrity of a non-pipelined transaction requires
#                        waiting until the read(s) in progress are finished).
# socketRdQueue($connId) List of tokens that are queued for reading later.
# socketWrQueue($connId) List of tokens that are queued for writing later.
# socketPhQueue($sock)   List of tokens that are queued to use a placeholder
#                        socket, when the real socket has not yet been created.
# socketClosing($connId) (boolean) true iff a server response header indicates
#                        that the server will close the connection at the end of
#                        the current response.
# socketPlayCmd($connId) The command to execute to replay pending and
#                        part-completed transactions if the socket closes early.
# socketCoEvent($connId) Identifier for the "after idle" event that will launch
#                        an OpenSocket coroutine to open or re-use a socket.
# socketProxyId($connId) The type of proxy that this socket uses: values are
#                        those of state(proxyUsed) i.e. none, HttpProxy,
#                        SecureProxy, and SecureProxyFailed.
#                        The value is not used for anything by http, its purpose
#                        is to set the value of state() for caller information.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Using socketWrState(*), socketWrQueue(*), socketRdState(*), socketRdQueue(*)
# ------------------------------------------------------------------------------
# The element socketWrState($connId) has a value which is either the name of
# the token that is permitted to write to the socket, or "Wready" if no
# token is permitted to write.
#
# The code that sets the value to Wready immediately calls
# http::NextPipelinedWrite, which examines socketWrQueue($connId) and
# processes the next request in the queue, if there is one.  The value
# Wready is not found when the interpreter is in the event loop unless the
# socket is idle.
#
# The element socketRdState($connId) has a value which is either the name of
# the token that is permitted to read from the socket, or "Rready" if no
# token is permitted to read.
#
# The code that sets the value to Rready then examines
# socketRdQueue($connId) and processes the next request in the queue, if
# there is one.  The value Rready is not found when the interpreter is in
# the event loop unless the socket is idle.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#  Proc http::ScheduleRequest
# ------------------------------------------------------------------------------
# Command to either begin the HTTP request, or add it to the appropriate queue.
# Called from two places in ConfigureNewSocket.
#
# Arguments:
# token         - connection token (name of an array)
#
# Return Value: none
# ------------------------------------------------------------------------------

proc http::ScheduleRequest {token} {
    variable $token
    upvar 0 $token state
    set tk [namespace tail $token]

    Log >L$tk ScheduleRequest

    variable socketMapping
    variable socketRdState
    variable socketWrState
    variable socketRdQueue
    variable socketWrQueue
    variable socketPhQueue
    variable socketClosing
    variable socketPlayCmd
    variable socketCoEvent
    variable socketProxyId

    set Unfinished 0

    set reusing $state(reusing)
    set sockNew $state(sock)

    # The "if" tests below: must test against the current values of
    # socketWrState, socketRdState, and so the tests must be done here,
    # not earlier in PreparePersistentConnection.

    if {$state(alreadyQueued)} {
	# The request has been appended to the queue of a persistent socket
	# (that is scheduled to close and have its queue replayed).
	#
	# A write may or may not be in progress.  There is no need to set
	# socketWrState to prevent another call stealing write access - all
	# subsequent calls on this socket will come here because the socket
	# will close after the current read, and its
	# socketClosing($connId) is 1.
	##Log "HTTP request for token $token is queued"

    } elseif {    $reusing
	       && $state(-pipeline)
	       && ($socketWrState($state(socketinfo)) ne "Wready")
    } {
	##Log "HTTP request for token $token is queued for pipelined use"
	lappend socketWrQueue($state(socketinfo)) $token

    } elseif {    $reusing
	       && (!$state(-pipeline))
	       && ($socketWrState($state(socketinfo)) ne "Wready")
    } {
	# A write is queued or in progress.  Lappend to the write queue.
	##Log "HTTP request for token $token is queued for nonpipeline use"
	lappend socketWrQueue($state(socketinfo)) $token

    } elseif {    $reusing
	       && (!$state(-pipeline))
	       && ($socketWrState($state(socketinfo)) eq "Wready")
	       && ($socketRdState($state(socketinfo)) ne "Rready")
    } {
	# A read is queued or in progress, but not a write.  Cannot start the
	# nonpipeline transaction, but must set socketWrState to prevent a
	# pipelined request jumping the queue.
	##Log "HTTP request for token $token is queued for nonpipeline use"
	#Log re-use nonpipeline, GRANT delayed write access to $token in geturl
	set socketWrState($state(socketinfo)) peNding
	lappend socketWrQueue($state(socketinfo)) $token

    } else {
	if {$reusing && $state(-pipeline)} {
	    #Log new, init for pipelined, GRANT write access to $token in geturl
	    # DO NOT grant premature read access to the socket.
	    # set socketRdState($state(socketinfo)) $token
	    set socketWrState($state(socketinfo)) $token
	} elseif {$reusing} {
	    # socketWrState is not used by this non-pipelined transaction.
	    # We cannot leave it as "Wready" because the next call to
	    # http::geturl with a pipelined transaction would conclude that the
	    # socket is available for writing.
	    #Log new, init for nonpipeline, GRANT r/w access to $token in geturl
	    set socketRdState($state(socketinfo)) $token
	    set socketWrState($state(socketinfo)) $token
	}

	# Process the request now.
	# - Command is not called unless $state(sock) is a real socket handle
	#   and not a placeholder.
	# - All (!$reusing) cases come here.
	# - Some $reusing cases come here too if the connection is
	#   marked as ready.  Those $reusing cases are:
	#   $reusing && ($socketWrState($state(socketinfo)) eq "Wready") &&
	#   EITHER !$pipeline && ($socketRdState($state(socketinfo)) eq "Rready")
	#   OR      $pipeline
	#
	#Log ---- $state(socketinfo) << conn to $token for HTTP request (a)
	##Log "  ScheduleRequest" $token -- fileevent $state(sock) writable for $token
	# Connect does its own fconfigure.

	lassign $state(connArgs) proto phost srvurl

	if {[catch {
		fileevent $state(sock) writable \
			[list http::Connect $token $proto $phost $srvurl]
	} res opts]} {
	    # The socket no longer exists.
	    ##Log bug -- socket gone -- $res -- $opts
	}

    }

    return
}


# ------------------------------------------------------------------------------
#  Proc http::SendHeader
# ------------------------------------------------------------------------------
# Command to send a request header, and keep a copy in state(requestHeaders)
# for debugging purposes.
#
# Arguments:
# token       - connection token (name of an array)
# key         - header name
# value       - header value
#
# Return Value: none
# ------------------------------------------------------------------------------

proc http::SendHeader {token key value} {
    variable $token
    upvar 0 $token state
    set tk [namespace tail $token]
    set sock $state(sock)
    lappend state(requestHeaders) [string tolower $key] $value
    puts $sock "$key: $value"
    return
}

# http::Connected --
#
#	Callback used when the connection to the HTTP server is actually
#	established.
#
# Arguments:
#	token	State token.
#	proto	What protocol (http, https, etc.) was used to connect.
#	phost	Are we using keep-alive? Non-empty if yes.
#	srvurl	Service-local URL that we're requesting
# Results:
#	None.

proc http::Connected {token proto phost srvurl} {
    variable http
    variable urlTypes
    variable socketMapping
    variable socketRdState
    variable socketWrState
    variable socketRdQueue
    variable socketWrQueue
    variable socketPhQueue
    variable socketClosing
    variable socketPlayCmd
    variable socketCoEvent
    variable socketProxyId

    variable $token
    upvar 0 $token state
    set tk [namespace tail $token]

    if {$state(reusing) && (!$state(-pipeline)) && ($state(-timeout) > 0)} {
	set state(after) [after $state(-timeout) \
		[list http::reset $token timeout]]
    }

    # Set back the variables needed here.
    set sock $state(sock)
    set isQueryChannel [info exists state(-querychannel)]
    set isQuery [info exists state(-query)]
    regexp {^(.+):([^:]+)$} $state(socketinfo) {} host port

    set lower [string tolower $proto]
    set defport [lindex $urlTypes($lower) 0]

    # Send data in cr-lf format, but accept any line terminators.
    # Initialisation to {auto *} now done in geturl, KeepSocket and DoneRequest.
    # We are concerned here with the request (write) not the response (read).
    lassign [fconfigure $sock -translation] trRead trWrite
    fconfigure $sock -translation [list $trRead crlf] \
		     -buffersize $state(-blocksize)
    if {[package vsatisfies [package provide Tcl] 9.0-]} {
	fconfigure $sock -profile replace
    }

    # The following is disallowed in safe interpreters, but the socket is
    # already in non-blocking mode in that case.

    catch {fconfigure $sock -blocking off}
    set how GET
    if {$isQuery} {
	set state(querylength) [string length $state(-query)]
	if {$state(querylength) > 0} {
	    set how POST
	    set contDone 0
	} else {
	    # There's no query data.
	    unset state(-query)
	    set isQuery 0
	}
    } elseif {$state(-validate)} {
	set how HEAD
    } elseif {$isQueryChannel} {
	set how POST
	# The query channel must be blocking for the async Write to
	# work properly.
	fconfigure $state(-querychannel) -blocking 1 -translation binary
	set contDone 0
    }
    if {[info exists state(-method)] && ($state(-method) ne "")} {
	set how $state(-method)
    }
    set accept_types_seen 0

    Log ^B$tk begin sending request - token $token

    if {[catch {
	if {[info exists state(bypass)]} {
	    set state(method) [lindex [split $state(bypass) { }] 0]
	    set state(requestHeaders) {}
	    set state(requestLine) $state(bypass)
	} else {
	    set state(method) $how
	    set state(requestHeaders) {}
	    set state(requestLine) "$how $srvurl HTTP/$state(-protocol)"
	}
	puts $sock $state(requestLine)
	set hostValue [GetFieldValue $state(-headers) Host]
	if {$hostValue ne {}} {
	    # Allow Host spoofing. [Bug 928154]
	    regexp {^[^:]+} $hostValue state(host)
	    SendHeader $token Host $hostValue
	} elseif {$port == $defport} {
	    # Don't add port in this case, to handle broken servers. [Bug
	    # #504508]
	    set state(host) $host
	    SendHeader $token Host $host
	} else {
	    set state(host) $host
	    SendHeader $token Host "$host:$port"
	}
	SendHeader $token User-Agent $http(-useragent)
	if {($state(-protocol) > 1.0) && $state(-keepalive)} {
	    # Send this header, because a 1.1 server is not compelled to treat
	    # this as the default.
	    set ConnVal keep-alive
	} elseif {($state(-protocol) > 1.0)} {
	    # RFC2616 sec 8.1.2.1
	    set ConnVal close
	} else {
	    # ($state(-protocol) <= 1.0)
	    # RFC7230 A.1
	    # Some server implementations of HTTP/1.0 have a faulty
	    # implementation of RFC 2068 Keep-Alive.
	    # Don't leave this to chance.
	    # For HTTP/1.0 we have already "set state(connection) close"
	    # and "state(-keepalive) 0".
	    set ConnVal close
	}
	# Proxy authorisation (cf. mod by Anders Ramdahl to autoproxy by
	# Pat Thoyts).
	if {($http(-proxyauth) ne {}) && ($state(proxyUsed) eq {HttpProxy})} {
	    SendHeader $token Proxy-Authorization $http(-proxyauth)
	}
	# RFC7230 A.1 - "clients are encouraged not to send the
	# Proxy-Connection header field in any requests"
	set accept_encoding_seen 0
	set content_type_seen 0
	set connection_seen 0
	foreach {key value} $state(-headers) {
	    set value [string map [list \n "" \r ""] $value]
	    set key [string map {" " -} [string trim $key]]
	    if {[string equal -nocase $key "host"]} {
		continue
	    }
	    if {[string equal -nocase $key "accept-encoding"]} {
		set accept_encoding_seen 1
	    }
	    if {[string equal -nocase $key "accept"]} {
		set accept_types_seen 1
	    }
	    if {[string equal -nocase $key "content-type"]} {
		set content_type_seen 1
	    }
	    if {[string equal -nocase $key "content-length"]} {
		set contDone 1
		set state(querylength) $value
	    }
	    if {    [string equal -nocase $key "connection"]
		    && [info exists state(bypass)]
	    } {
		# Value supplied in -headers overrides $ConnVal.
		set connection_seen 1
	    } elseif {[string equal -nocase $key "connection"]} {
		# Remove "close" or "keep-alive" and use our own value.
		# In an upgrade request, the upgrade is not guaranteed.
		# Value "close" or "keep-alive" tells the server what to do
		# if it refuses the upgrade.  We send a single "Connection"
		# header because some websocket servers, e.g. civetweb, reject
		# multiple headers. Bug [d01de3281f] of tcllib/websocket.
		set connection_seen 1
		set listVal $state(connectionValues)
		if {[set pos [lsearch $listVal close]] != -1} {
		    set listVal [lreplace $listVal $pos $pos]
		}
		if {[set pos [lsearch $listVal keep-alive]] != -1} {
		    set listVal [lreplace $listVal $pos $pos]
		}
		lappend listVal $ConnVal
		set value [join $listVal {, }]
	    }
	    if {[string length $key]} {
		SendHeader $token $key $value
	    }
	}
	# Allow overriding the Accept header on a per-connection basis. Useful
	# for working with REST services. [Bug c11a51c482]
	if {!$accept_types_seen} {
	    SendHeader $token Accept $state(accept-types)
	}
	if {    (!$accept_encoding_seen)
	     && (![info exists state(-handler)])
	     && $http(-zip)
	} {
	    SendHeader $token Accept-Encoding gzip,deflate
	} elseif {!$accept_encoding_seen} {
	    SendHeader $token Accept-Encoding identity
	}
	if {!$connection_seen} {
	    SendHeader $token Connection $ConnVal
	}
	if {$isQueryChannel && ($state(querylength) == 0)} {
	    # Try to determine size of data in channel. If we cannot seek, the
	    # surrounding catch will trap us

	    set start [tell $state(-querychannel)]
	    seek $state(-querychannel) 0 end
	    set state(querylength) \
		    [expr {[tell $state(-querychannel)] - $start}]
	    seek $state(-querychannel) $start
	}

	# Note that we don't do Cookie2; that's much nastier and not normally
	# observed in practice either. It also doesn't fix the multitude of
	# bugs in the basic cookie spec.
	if {$http(-cookiejar) ne ""} {
	    set cookies ""
	    set separator ""
	    foreach {key value} [{*}$http(-cookiejar) \
		    getCookies $proto $host $state(path)] {
		append cookies $separator $key = $value
		set separator "; "
	    }
	    if {$cookies ne ""} {
		SendHeader $token Cookie $cookies
	    }
	}

	# Flush the request header and set up the fileevent that will either
	# push the POST data or read the response.
	#
	# fileevent note:
	#
	# It is possible to have both the read and write fileevents active at
	# this point. The only scenario it seems to affect is a server that
	# closes the connection without reading the POST data. (e.g., early
	# versions TclHttpd in various error cases). Depending on the
	# platform, the client may or may not be able to get the response from
	# the server because of the error it will get trying to write the post
	# data. Having both fileevents active changes the timing and the
	# behavior, but no two platforms (among Solaris, Linux, and NT) behave
	# the same, and none behave all that well in any case. Servers should
	# always read their POST data if they expect the client to read their
	# response.

	if {$isQuery || $isQueryChannel} {
	    # POST method.
	    if {!$content_type_seen} {
		SendHeader $token Content-Type $state(-type)
	    }
	    if {!$contDone} {
		SendHeader $token Content-Length $state(querylength)
	    }
	    puts $sock ""
	    flush $sock
	    # Flush flushes the error in the https case with a bad handshake:
	    # else the socket never becomes writable again, and hangs until
	    # timeout (if any).

	    lassign [fconfigure $sock -translation] trRead trWrite
	    fconfigure $sock -translation [list $trRead binary]
	    fileevent $sock writable [list http::Write $token]
	    # The http::Write command decides when to make the socket readable,
	    # using the same test as the GET/HEAD case below.
	} else {
	    # GET or HEAD method.
	    if {    (![catch {fileevent $sock readable} binding])
		 && ($binding eq [list http::CheckEof $sock])
	    } {
		# Remove the "fileevent readable" binding of an idle persistent
		# socket to http::CheckEof.  We can no longer treat bytes
		# received as junk. The server might still time out and
		# half-close the socket if it has not yet received the first
		# "puts".
		fileevent $sock readable {}
	    }
	    puts $sock ""
	    flush $sock
	    Log ^C$tk end sending request - token $token
	    # End of writing (GET/HEAD methods).  The request has been sent.

	    DoneRequest $token
	}

    } err]} {
	# The socket probably was never connected, OR the connection dropped
	# later, OR https handshake error, which may be discovered as late as
	# the "flush" command above...
	Log "WARNING - if testing, pay special attention to this\
		case (GI) which is seldom executed - token $token"
	if {[info exists state(reusing)] && $state(reusing)} {
	    # The socket was closed at the server end, and closed at
	    # this end by http::CheckEof.
	    if {[TestForReplay $token write $err a]} {
		return
	    } else {
		Finish $token {failed to re-use socket}
	    }

	    # else:
	    # This is NOT a persistent socket that has been closed since its
	    # last use.
	    # If any other requests are in flight or pipelined/queued, they will
	    # be discarded.
	} elseif {$state(status) eq ""} {
	    # https handshake errors come here, for
	    # Tcl 9.0 without http::SecureProxyConnect, and for Tcl 8.6.
	    set msg [registerError $sock]
	    registerError $sock {}
	    if {$msg eq {}} {
		set msg {failed to use socket}
	    }
	    Finish $token $msg
	} elseif {$state(status) ne "error"} {
	    Finish $token $err
	}
    }
    return
}

# http::registerError
#
#	Called (for example when processing TclTLS activity) to register
#	an error for a connection on a specific socket.  This helps
#	http::Connected to deliver meaningful error messages, e.g. when a TLS
#	certificate fails verification.
#
#	Usage: http::registerError socket ?newValue?
#
#	"set" semantics, except that a "get" (a call without a new value) for a
#	non-existent socket returns {}, not an error.

proc http::registerError {sock args} {
    variable registeredErrors

    if {    ([llength $args] == 0)
	 && (![info exists registeredErrors($sock)])
    } {
	return
    } elseif {    ([llength $args] == 1)
	       && ([lindex $args 0] eq {})
    } {
	unset -nocomplain registeredErrors($sock)
	return
    }
    set registeredErrors($sock) {*}$args
}

# http::DoneRequest --
#
#	Command called when a request has been sent.  It will arrange the
#	next request and/or response as appropriate.
#
#	If this command is called when $socketClosing(*), the request $token
#	that calls it must be pipelined and destined to fail.

proc http::DoneRequest {token} {
    variable http
    variable socketMapping
    variable socketRdState
    variable socketWrState
    variable socketRdQueue
    variable socketWrQueue
    variable socketPhQueue
    variable socketClosing
    variable socketPlayCmd
    variable socketCoEvent
    variable socketProxyId

    variable $token
    upvar 0 $token state
    set tk [namespace tail $token]
    set sock $state(sock)

    # If pipelined, connect the next HTTP request to the socket.
    if {$state(reusing) && $state(-pipeline)} {
	# Enable next token (if any) to write.
	# The value "Wready" is set only here, and
	# in http::Event after reading the response-headers of a
	# non-reusing transaction.
	# Previous value is $token. It cannot be pending.
	set socketWrState($state(socketinfo)) Wready

	# Now ready to write the next pipelined request (if any).
	http::NextPipelinedWrite $token
    } else {
	# If pipelined, this is the first transaction on this socket.  We wait
	# for the response headers to discover whether the connection is
	# persistent.  (If this is not done and the connection is not
	# persistent, we SHOULD retry and then MUST NOT pipeline before knowing
	# that we have a persistent connection
	# (rfc2616 8.1.2.2)).
    }

    # Connect to receive the response, unless the socket is pipelined
    # and another response is being sent.
    # This code block is separate from the code below because there are
    # cases where socketRdState already has the value $token.
    if {    $state(-keepalive)
	 && $state(-pipeline)
	 && [info exists socketRdState($state(socketinfo))]
	 && ($socketRdState($state(socketinfo)) eq "Rready")
    } {
	#Log pipelined, GRANT read access to $token in Connected
	set socketRdState($state(socketinfo)) $token
    }

    if {    $state(-keepalive)
	 && $state(-pipeline)
	 && [info exists socketRdState($state(socketinfo))]
	 && ($socketRdState($state(socketinfo)) ne $token)
    } {
	# Do not read from the socket until it is ready.
	##Log "HTTP response for token $token is queued for pipelined use"
	# If $socketClosing(*), then the caller will be a pipelined write and
	# execution will come here.
	# This token has already been recorded as "in flight" for writing.
	# When the socket is closed, the read queue will be cleared in
	# CloseQueuedQueries and so the "lappend" here has no effect.
	lappend socketRdQueue($state(socketinfo)) $token
    } else {
	# In the pipelined case, connection for reading depends on the
	# value of socketRdState.
	# In the nonpipeline case, connection for reading always occurs.
	ReceiveResponse $token
    }
    return
}

# http::ReceiveResponse
#
#	Connects token to its socket for reading.

proc http::ReceiveResponse {token} {
    variable $token
    upvar 0 $token state
    set tk [namespace tail $token]
    set sock $state(sock)

    #Log ---- $state(socketinfo) >> conn to $token for HTTP response
    lassign [fconfigure $sock -translation] trRead trWrite
    fconfigure $sock -translation [list auto $trWrite] \
		     -buffersize $state(-blocksize)
    if {[package vsatisfies [package provide Tcl] 9.0-]} {
	fconfigure $sock -profile replace
    }
    Log ^D$tk begin receiving response - token $token

    coroutine ${token}--EventCoroutine http::Event $sock $token
    if {[info exists state(-handler)] || [info exists state(-progress)]} {
	fileevent $sock readable [list http::EventGateway $sock $token]
    } else {
	fileevent $sock readable ${token}--EventCoroutine
    }
    return
}


# http::EventGateway
#
#	Bug [c2dc1da315].
#	- Recursive launch of the coroutine can occur if a -handler or -progress
#	  callback is used, and the callback command enters the event loop.
#	- To prevent this, the fileevent "binding" is disabled while the
#	  coroutine is in flight.
#	- If a recursive call occurs despite these precautions, it is not
#	  trapped and discarded here, because it is better to report it as a
#	  bug.
#	- Although this solution is believed to be sufficiently general, it is
#	  used only if -handler or -progress is specified.  In other cases,
#	  the coroutine is called directly.

proc http::EventGateway {sock token} {
    variable $token
    upvar 0 $token state
    fileevent $sock readable {}
    catch {${token}--EventCoroutine} res opts
    if {[info commands ${token}--EventCoroutine] ne {}} {
	# The coroutine can be deleted by completion (a non-yield return), by
	# http::Finish (when there is a premature end to the transaction), by
	# http::reset or http::cleanup, or if the caller set option -channel
	# but not option -handler: in the last case reading from the socket is
	# now managed by commands ::http::Copy*, http::ReceiveChunked, and
	# http::MakeTransformationChunked.
	#
	# Catch in case the coroutine has closed the socket.
	catch {fileevent $sock readable [list http::EventGateway $sock $token]}
    }

    # If there was an error, re-throw it.
    return -options $opts $res
}


# http::NextPipelinedWrite
#
# - Connecting a socket to a token for writing is done by this command and by
#   command KeepSocket.
# - If another request has a pipelined write scheduled for $token's socket,
#   and if the socket is ready to accept it, connect the write and update
#   the queue accordingly.
# - This command is called from http::DoneRequest and http::Event,
#   IF $state(-pipeline) AND (the current transfer has reached the point at
#   which the socket is ready for the next request to be written).
# - This command is called when a token has write access and is pipelined and
#   keep-alive, and sets socketWrState to Wready.
# - The command need not consider the case where socketWrState is set to a token
#   that does not yet have write access.  Such a token is waiting for Rready,
#   and the assignment of the connection to the token will be done elsewhere (in
#   http::KeepSocket).
# - This command cannot be called after socketWrState has been set to a
#   "pending" token value (that is then overwritten by the caller), because that
#   value is set by this command when it is called by an earlier token when it
#   relinquishes its write access, and the pending token is always the next in
#   line to write.

proc http::NextPipelinedWrite {token} {
    variable http
    variable socketRdState
    variable socketWrState
    variable socketWrQueue
    variable socketClosing
    variable $token
    upvar 0 $token state
    set connId $state(socketinfo)

    if {    [info exists socketClosing($connId)]
	 && $socketClosing($connId)
    } {
	# socketClosing(*) is set because the server has sent a
	# "Connection: close" header.
	# Behave as if the queues are empty - so do nothing.
    } elseif {    $state(-pipeline)
	 && [info exists socketWrState($connId)]
	 && ($socketWrState($connId) eq "Wready")

	 && [info exists socketWrQueue($connId)]
	 && [llength $socketWrQueue($connId)]
	 && ([set token2 [lindex $socketWrQueue($connId) 0]
	      set ${token2}(-pipeline)
	     ]
	    )
    } {
	# - The usual case for a pipelined connection, ready for a new request.
	#Log pipelined, GRANT write access to $token2 in NextPipelinedWrite
	set conn [set ${token2}(connArgs)]
	set socketWrState($connId) $token2
	set socketWrQueue($connId) [lrange $socketWrQueue($connId) 1 end]
	# Connect does its own fconfigure.
	fileevent $state(sock) writable [list http::Connect $token2 {*}$conn]
	#Log ---- $connId << conn to $token2 for HTTP request (b)

	# In the tests below, the next request will be nonpipeline.
    } elseif {    $state(-pipeline)
	       && [info exists socketWrState($connId)]
	       && ($socketWrState($connId) eq "Wready")

	       && [info exists socketWrQueue($connId)]
	       && [llength $socketWrQueue($connId)]
	       && (![ set token3 [lindex $socketWrQueue($connId) 0]
		      set ${token3}(-pipeline)
		    ]
		  )

	       && [info exists socketRdState($connId)]
	       && ($socketRdState($connId) eq "Rready")
    } {
	# The case in which the next request will be non-pipelined, and the read
	# and write queues is ready: which is the condition for a non-pipelined
	# write.
	set conn [set ${token3}(connArgs)]
	#Log nonpipeline, GRANT r/w access to $token3 in NextPipelinedWrite
	set socketRdState($connId) $token3
	set socketWrState($connId) $token3
	set socketWrQueue($connId) [lrange $socketWrQueue($connId) 1 end]
	# Connect does its own fconfigure.
	fileevent $state(sock) writable [list http::Connect $token3 {*}$conn]
	#Log ---- $state(sock) << conn to $token3 for HTTP request (c)

    } elseif {    $state(-pipeline)
	 && [info exists socketWrState($connId)]
	 && ($socketWrState($connId) eq "Wready")

	 && [info exists socketWrQueue($connId)]
	 && [llength $socketWrQueue($connId)]
	 && (![set token2 [lindex $socketWrQueue($connId) 0]
	      set ${token2}(-pipeline)
	     ]
	    )
    } {
	# - The case in which the next request will be non-pipelined, but the
	#   read queue is NOT ready.
	# - A read is queued or in progress, but not a write.  Cannot start the
	#   nonpipeline transaction, but must set socketWrState to prevent a new
	#   pipelined request (in http::geturl) jumping the queue.
	# - Because socketWrState($connId) is not set to Wready, the assignment
	#   of the connection to $token2 will be done elsewhere - by command
	#   http::KeepSocket when $socketRdState($connId) is set to "Rready".

	#Log re-use nonpipeline, GRANT delayed write access to $token in NextP..
	set socketWrState($connId) peNding
    }
    return
}

# http::CancelReadPipeline
#
#	Cancel pipelined responses on a closing "Keep-Alive" socket.
#
#	- Called by a variable trace on "unset socketRdState($connId)".
#	- The variable relates to a Keep-Alive socket, which has been closed.
#	- Cancels all pipelined responses. The requests have been sent,
#	  the responses have not yet been received.
#	- This is a hard cancel that ends each transaction with error status,
#	  and closes the connection. Do not use it if you want to replay failed
#	  transactions.
#	- N.B. Always delete ::http::socketRdState($connId) before deleting
#	  ::http::socketRdQueue($connId), or this command will do nothing.
#
# Arguments
#	As for a trace command on a variable.

proc http::CancelReadPipeline {name1 connId op} {
    variable socketRdQueue
    ##Log CancelReadPipeline $name1 $connId $op
    if {[info exists socketRdQueue($connId)]} {
	set msg {the connection was closed by CancelReadPipeline}
	foreach token $socketRdQueue($connId) {
	    set tk [namespace tail $token]
	    Log ^X$tk end of response "($msg)" - token $token
	    set ${token}(status) eof
	    Finish $token ;#$msg
	}
	set socketRdQueue($connId) {}
    }
    return
}

# http::CancelWritePipeline
#
#	Cancel queued events on a closing "Keep-Alive" socket.
#
#	- Called by a variable trace on "unset socketWrState($connId)".
#	- The variable relates to a Keep-Alive socket, which has been closed.
#	- In pipelined or nonpipeline case: cancels all queued requests.  The
#	  requests have not yet been sent, the responses are not due.
#	- This is a hard cancel that ends each transaction with error status,
#	  and closes the connection. Do not use it if you want to replay failed
#	  transactions.
#	- N.B. Always delete ::http::socketWrState($connId) before deleting
#	  ::http::socketWrQueue($connId), or this command will do nothing.
#
# Arguments
#	As for a trace command on a variable.

proc http::CancelWritePipeline {name1 connId op} {
    variable socketWrQueue

    ##Log CancelWritePipeline $name1 $connId $op
    if {[info exists socketWrQueue($connId)]} {
	set msg {the connection was closed by CancelWritePipeline}
	foreach token $socketWrQueue($connId) {
	    set tk [namespace tail $token]
	    Log ^X$tk end of response "($msg)" - token $token
	    set ${token}(status) eof
	    Finish $token ;#$msg
	}
	set socketWrQueue($connId) {}
    }
    return
}

# http::ReplayIfDead --
#
# - A query on a re-used persistent socket failed at the earliest opportunity,
#   because the socket had been closed by the server.  Keep the token, tidy up,
#   and try to connect on a fresh socket.
# - The connection is monitored for eof by the command http::CheckEof.  Thus
#   http::ReplayIfDead is needed only when a server event (half-closing an
#   apparently idle connection), and a client event (sending a request) occur at
#   almost the same time, and neither client nor server detects the other's
#   action before performing its own (an "asynchronous close event").
# - To simplify testing of http::ReplayIfDead, set TEST_EOF 1 in
#   http::KeepSocket, and then http::ReplayIfDead will be called if http::geturl
#   is called at any time after the server timeout.
#
# Arguments:
#	token	Connection token.
#
# Side Effects:
#	Use the same token, but try to open a new socket.

proc http::ReplayIfDead {token doing} {
    variable socketMapping
    variable socketRdState
    variable socketWrState
    variable socketRdQueue
    variable socketWrQueue
    variable socketPhQueue
    variable socketClosing
    variable socketPlayCmd
    variable socketCoEvent
    variable socketProxyId

    variable $token
    upvar 0 $token state

    Log running http::ReplayIfDead for $token $doing

    # 1. Merge the tokens for transactions in flight, the read (response) queue,
    #    and the write (request) queue.

    set InFlightR {}
    set InFlightW {}

    # Obtain the tokens for transactions in flight.
    if {$state(-pipeline)} {
	# Two transactions may be in flight.  The "read" transaction was first.
	# It is unlikely that the server would close the socket if a response
	# was pending; however, an earlier request (as well as the present
	# request) may have been sent and ignored if the socket was half-closed
	# by the server.

	if {    [info exists socketRdState($state(socketinfo))]
	     && ($socketRdState($state(socketinfo)) ne "Rready")
	} {
	    lappend InFlightR $socketRdState($state(socketinfo))
	} elseif {($doing eq "read")} {
	    lappend InFlightR $token
	}

	if {    [info exists socketWrState($state(socketinfo))]
	     && $socketWrState($state(socketinfo)) ni {Wready peNding}
	} {
	    lappend InFlightW $socketWrState($state(socketinfo))
	} elseif {($doing eq "write")} {
	    lappend InFlightW $token
	}

	# Report any inconsistency of $token with socket*state.
	if {    ($doing eq "read")
	     && [info exists socketRdState($state(socketinfo))]
	     && ($token ne $socketRdState($state(socketinfo)))
	} {
	    Log WARNING - ReplayIfDead pipelined token $token $doing \
		    ne socketRdState($state(socketinfo)) \
		      $socketRdState($state(socketinfo))

	} elseif {
		($doing eq "write")
	     && [info exists socketWrState($state(socketinfo))]
	     && ($token ne $socketWrState($state(socketinfo)))
	} {
	    Log WARNING - ReplayIfDead pipelined token $token $doing \
		    ne socketWrState($state(socketinfo)) \
		      $socketWrState($state(socketinfo))
	}
    } else {
	# One transaction should be in flight.
	# socketRdState, socketWrQueue are used.
	# socketRdQueue should be empty.

	# Report any inconsistency of $token with socket*state.
	if {$token ne $socketRdState($state(socketinfo))} {
	    Log WARNING - ReplayIfDead nonpipeline token $token $doing \
		    ne socketRdState($state(socketinfo)) \
		      $socketRdState($state(socketinfo))
	}

	# Report the inconsistency that socketRdQueue is non-empty.
	if {    [info exists socketRdQueue($state(socketinfo))]
	     && ($socketRdQueue($state(socketinfo)) ne {})
	} {
	    Log WARNING - ReplayIfDead nonpipeline token $token $doing \
		    has read queue socketRdQueue($state(socketinfo)) \
		    $socketRdQueue($state(socketinfo)) ne {}
	}

	lappend InFlightW $socketRdState($state(socketinfo))
	set socketRdQueue($state(socketinfo)) {}
    }

    set newQueue {}
    lappend newQueue {*}$InFlightR
    lappend newQueue {*}$socketRdQueue($state(socketinfo))
    lappend newQueue {*}$InFlightW
    lappend newQueue {*}$socketWrQueue($state(socketinfo))


    # 2. Tidy up token.  This is a cut-down form of Finish/CloseSocket.
    #    Do not change state(status).
    #    No need to after cancel state(after) - either this is done in
    #    ReplayCore/ReInit, or Finish is called.

    catch {close $state(sock)}
    Unset $state(socketinfo)

    # 2a. Tidy the tokens in the queues - this is done in ReplayCore/ReInit.
    # - Transactions, if any, that are awaiting responses cannot be completed.
    #   They are listed for re-sending in newQueue.
    # - All tokens are preserved for re-use by ReplayCore, and their variables
    #   will be re-initialised by calls to ReInit.
    # - The relevant element of socketMapping, socketRdState, socketWrState,
    #   socketRdQueue, socketWrQueue, socketClosing, socketPlayCmd will be set
    #   to new values in ReplayCore.

    ReplayCore $newQueue
    return
}

# http::ReplayIfClose --
#
#	A request on a socket that was previously "Connection: keep-alive" has
#	received a "Connection: close" response header.  The server supplies
#	that response correctly, but any later requests already queued on this
#	connection will be lost when the socket closes.
#
#	This command takes arguments that represent the socketWrState,
#	socketRdQueue and socketWrQueue for this connection.  The socketRdState
#	is not needed because the server responds in full to the request that
#	received the "Connection: close" response header.
#
#	Existing request tokens $token (::http::$n) are preserved.  The caller
#	will be unaware that the request was processed this way.

proc http::ReplayIfClose {Wstate Rqueue Wqueue} {
    Log running http::ReplayIfClose for $Wstate $Rqueue $Wqueue

    if {$Wstate in $Rqueue || $Wstate in $Wqueue} {
	Log WARNING duplicate token in http::ReplayIfClose - token $Wstate
	set Wstate Wready
    }

    # 1. Create newQueue
    set InFlightW {}
    if {$Wstate ni {Wready peNding}} {
	lappend InFlightW $Wstate
    }
    ##Log $Rqueue -- $InFlightW -- $Wqueue
    set newQueue {}
    lappend newQueue {*}$Rqueue
    lappend newQueue {*}$InFlightW
    lappend newQueue {*}$Wqueue

    # 2. Cleanup - none needed, done by the caller.

    ReplayCore $newQueue
    return
}

# http::ReInit --
#
#	Command to restore a token's state to a condition that
#	makes it ready to replay a request.
#
#	Command http::geturl stores extra state in state(tmp*) so
#	we don't need to do the argument processing again.
#
#	The caller must:
#	- Set state(reusing) and state(sock) to their new values after calling
#	  this command.
#	- Unset state(tmpState), state(tmpOpenCmd) if future calls to ReplayCore
#	  or ReInit are inappropriate for this token. Typically only one retry
#	  is allowed.
#	The caller may also unset state(tmpConnArgs) if this value (and the
#	token) will be used immediately.  The value is needed by tokens that
#	will be stored in a queue.
#
# Arguments:
#	token	Connection token.
#
# Return Value: (boolean) true iff the re-initialisation was successful.

proc http::ReInit {token} {
    variable $token
    upvar 0 $token state

    if {!(
	      [info exists state(tmpState)]
	   && [info exists state(tmpOpenCmd)]
	   && [info exists state(tmpConnArgs)]
	 )
    } {
	Log FAILED in http::ReInit via ReplayCore - NO tmp vars for $token
	return 0
    }

    if {[info exists state(after)]} {
	after cancel $state(after)
	unset state(after)
    }
    if {[info exists state(socketcoro)]} {
	Log $token Cancel socket after-idle event (ReInit)
	after cancel $state(socketcoro)
	unset state(socketcoro)
    }

    # Don't alter state(status) - this would trigger http::wait if it is in use.
    set tmpState    $state(tmpState)
    set tmpOpenCmd  $state(tmpOpenCmd)
    set tmpConnArgs $state(tmpConnArgs)
    foreach name [array names state] {
	if {$name ne "status"} {
	    unset state($name)
	}
    }

    # Don't alter state(status).
    # Restore state(tmp*) - the caller may decide to unset them.
    # Restore state(tmpConnArgs) which is needed for connection.
    # state(tmpState), state(tmpOpenCmd) are needed only for retries.

    dict unset tmpState status
    array set state $tmpState
    set state(tmpState)    $tmpState
    set state(tmpOpenCmd)  $tmpOpenCmd
    set state(tmpConnArgs) $tmpConnArgs

    return 1
}

# http::ReplayCore --
#
#	Command to replay a list of requests, using existing connection tokens.
#
#	Abstracted from http::geturl which stores extra state in state(tmp*) so
#	we don't need to do the argument processing again.
#
# Arguments:
#	newQueue	List of connection tokens.
#
# Side Effects:
#	Use existing tokens, but try to open a new socket.

proc http::ReplayCore {newQueue} {
    variable TmpSockCounter

    variable socketMapping
    variable socketRdState
    variable socketWrState
    variable socketRdQueue
    variable socketWrQueue
    variable socketPhQueue
    variable socketClosing
    variable socketPlayCmd
    variable socketCoEvent
    variable socketProxyId

    if {[llength $newQueue] == 0} {
	# Nothing to do.
	return
    }

    ##Log running ReplayCore for {*}$newQueue
    set newToken [lindex $newQueue 0]
    set newQueue [lrange $newQueue 1 end]

    # 3. Use newToken, and restore its values of state(*).  Do not restore
    #    elements tmp* - we try again only once.

    set token $newToken
    variable $token
    upvar 0 $token state

    if {![ReInit $token]} {
	Log FAILED in http::ReplayCore - NO tmp vars
	Log ReplayCore reject $token
	Finish $token {cannot send this request again}
	return
    }

    set tmpState    $state(tmpState)
    set tmpOpenCmd  $state(tmpOpenCmd)
    set tmpConnArgs $state(tmpConnArgs)
    unset state(tmpState)
    unset state(tmpOpenCmd)
    unset state(tmpConnArgs)

    set state(reusing) 0
    set state(ReusingPlaceholder) 0
    set state(alreadyQueued) 0
    Log ReplayCore replay $token

    # Give the socket a placeholder name before it is created.
    set sock HTTP_PLACEHOLDER_[incr TmpSockCounter]
    set state(sock) $sock

    # Move the $newQueue into the placeholder socket's socketPhQueue.
    set socketPhQueue($sock) {}
    foreach tok $newQueue {
	if {[ReInit $tok]} {
	    set ${tok}(reusing) 1
	    set ${tok}(sock) $sock
	    lappend socketPhQueue($sock) $tok
	    Log ReplayCore replay $tok
	} else {
	    Log ReplayCore reject $tok
	    set ${tok}(reusing) 1
	    set ${tok}(sock) NONE
	    Finish $tok {cannot send this request again}
	}
    }

    AsyncTransaction $token

    return
}

# Data access functions:
# Data - the URL data
# Status - the transaction status: ok, reset, eof, timeout, error
# Code - the HTTP transaction code, e.g., 200
# Size - the size of the URL data

proc http::responseBody {token} {
    variable $token
    upvar 0 $token state
    return $state(body)
}
proc http::status {token} {
    if {![info exists $token]} {
	return "error"
    }
    variable $token
    upvar 0 $token state
    return $state(status)
}
proc http::responseLine {token} {
    variable $token
    upvar 0 $token state
    return $state(http)
}
proc http::requestLine {token} {
    variable $token
    upvar 0 $token state
    return $state(requestLine)
}
proc http::responseCode {token} {
    variable $token
    upvar 0 $token state
    if {[regexp {[0-9]{3}} $state(http) numeric_code]} {
	return $numeric_code
    } else {
	return $state(http)
    }
}
proc http::size {token} {
    variable $token
    upvar 0 $token state
    return $state(currentsize)
}
proc http::requestHeaders {token args} {
    set lenny  [llength $args]
    if {$lenny > 1} {
	return -code error {usage: ::http::requestHeaders token ?headerName?}
    } else {
	return [Meta $token request {*}$args]
    }
}
proc http::responseHeaders {token args} {
    set lenny  [llength $args]
    if {$lenny > 1} {
	return -code error {usage: ::http::responseHeaders token ?headerName?}
    } else {
	return [Meta $token response {*}$args]
    }
}
proc http::requestHeaderValue {token header} {
    Meta $token request $header VALUE
}
proc http::responseHeaderValue {token header} {
    Meta $token response $header VALUE
}
proc http::Meta {token who args} {
    variable $token
    upvar 0 $token state

    if {$who eq {request}} {
	set whom requestHeaders
    } elseif {$who eq {response}} {
	set whom meta
    } else {
	return -code error {usage: ::http::Meta token request|response ?headerName ?VALUE??}
    }

    set header [string tolower [lindex $args 0]]
    set how    [string tolower [lindex $args 1]]
    set lenny  [llength $args]
    if {$lenny == 0} {
	return $state($whom)
    } elseif {($lenny > 2) || (($lenny == 2) && ($how ne {value}))} {
	return -code error {usage: ::http::Meta token request|response ?headerName ?VALUE??}
    } else {
	set result {}
	set combined {}
	foreach {key value} $state($whom) {
	    if {$key eq $header} {
		lappend result $key $value
		append combined $value {, }
	    }
	}
	if {$lenny == 1} {
	    return $result
	} else {
	    return [string range $combined 0 end-2]
	}
    }
}


# ------------------------------------------------------------------------------
#  Proc http::responseInfo
# ------------------------------------------------------------------------------
# Command to return a dictionary of the most useful metadata of a HTTP
# response.
#
# Arguments:
# token       - connection token (name of an array)
#
# Return Value: a dict. See man page http(n) for a description of each item.
# ------------------------------------------------------------------------------

proc http::responseInfo {token} {
    variable $token
    upvar 0 $token state
    set result {}
    foreach {key origin name} {
	stage                 STATE  state
	status                STATE  status
	responseCode          STATE  responseCode
	reasonPhrase          STATE  reasonPhrase
	contentType           STATE  type
	binary                STATE  binary
	redirection           RESP   location
	upgrade               STATE  upgrade
	error                 ERROR  -
	postError             STATE  posterror
	method                STATE  method
	charset               STATE  charset
	compression           STATE  coding
	httpRequest           STATE  -protocol
	httpResponse          STATE  httpResponse
	url                   STATE  url
	connectionRequest     REQ    connection
	connectionResponse    RESP   connection
	connectionActual      STATE  connection
	transferEncoding      STATE  transfer
	totalPost             STATE  querylength
	currentPost           STATE  queryoffset
	totalSize             STATE  totalsize
	currentSize           STATE  currentsize
	proxyUsed             STATE  proxyUsed
    } {
	if {$origin eq {STATE}} {
	    if {[info exists state($name)]} {
		dict set result $key $state($name)
	    } else {
		# Should never come here
		dict set result $key {}
	    }
	} elseif {$origin eq {REQ}} {
	    dict set result $key [requestHeaderValue $token $name]
	} elseif {$origin eq {RESP}} {
	    dict set result $key [responseHeaderValue $token $name]
	} elseif {$origin eq {ERROR}} {
	    # Don't flood the dict with data.  The command ::http::error is
	    # available.
	    if {[info exists state(error)]} {
		set msg [lindex $state(error) 0]
	    } else {
		set msg {}
	    }
	    dict set result $key $msg
	} else {
	    # Should never come here
	    dict set result $key {}
	}
    }
    return $result
}
proc http::error {token} {
    variable $token
    upvar 0 $token state
    if {[info exists state(error)]} {
	return $state(error)
    }
    return
}
proc http::postError {token} {
    variable $token
    upvar 0 $token state
    if {[info exists state(postErrorFull)]} {
	return $state(postErrorFull)
    }
    return
}

# http::cleanup
#
#	Garbage collect the state associated with a transaction
#
# Arguments
#	token	The token returned from http::geturl
#
# Side Effects
#	Unsets the state array.

proc http::cleanup {token} {
    variable $token
    upvar 0 $token state
    if {[info commands ${token}--EventCoroutine] ne {}} {
	rename ${token}--EventCoroutine {}
    }
    if {[info commands ${token}--SocketCoroutine] ne {}} {
	rename ${token}--SocketCoroutine {}
    }
    if {[info exists state(after)]} {
	after cancel $state(after)
	unset state(after)
    }
    if {[info exists state(socketcoro)]} {
	Log $token Cancel socket after-idle event (cleanup)
	after cancel $state(socketcoro)
	unset state(socketcoro)
    }
    if {[info exists state]} {
	unset state
    }
    return
}

# http::Connect
#
#	This callback is made when an asynchronous connection completes.
#
# Arguments
#	token	The token returned from http::geturl
#
# Side Effects
#	Sets the status of the connection, which unblocks
#	the waiting geturl call

proc http::Connect {token proto phost srvurl} {
    variable $token
    upvar 0 $token state
    set tk [namespace tail $token]

    if {[catch {eof $state(sock)} tmp] || $tmp} {
	set err "due to unexpected EOF"
    } elseif {[set err [fconfigure $state(sock) -error]] ne ""} {
	# set err is done in test
    } else {
	# All OK
	set state(state) connecting
	fileevent $state(sock) writable {}
	::http::Connected $token $proto $phost $srvurl
	return
    }

    # Error cases.
	Log "WARNING - if testing, pay special attention to this\
		case (GJ) which is seldom executed - token $token"
	if {[info exists state(reusing)] && $state(reusing)} {
	    # The socket was closed at the server end, and closed at
	    # this end by http::CheckEof.
	    if {[TestForReplay $token write $err b]} {
		return
	    }

	    # else:
	    # This is NOT a persistent socket that has been closed since its
	    # last use.
	    # If any other requests are in flight or pipelined/queued, they will
	    # be discarded.
	}
	Finish $token "connect failed: $err"
    return
}

# http::Write
#
#	Write POST query data to the socket
#
# Arguments
#	token	The token for the connection
#
# Side Effects
#	Write the socket and handle callbacks.

proc http::Write {token} {
    variable http
    variable socketMapping
    variable socketRdState
    variable socketWrState
    variable socketRdQueue
    variable socketWrQueue
    variable socketPhQueue
    variable socketClosing
    variable socketPlayCmd
    variable socketCoEvent
    variable socketProxyId

    variable $token
    upvar 0 $token state
    set tk [namespace tail $token]
    set sock $state(sock)

    # Output a block.  Tcl will buffer this if the socket blocks
    set done 0
    if {[catch {
	# Catch I/O errors on dead sockets

	if {[info exists state(-query)]} {
	    # Chop up large query strings so queryprogress callback can give
	    # smooth feedback.
	    if {    $state(queryoffset) + $state(-queryblocksize)
		 >= $state(querylength)
	    } {
		# This will be the last puts for the request-body.
		if {    (![catch {fileevent $sock readable} binding])
		     && ($binding eq [list http::CheckEof $sock])
		} {
		    # Remove the "fileevent readable" binding of an idle
		    # persistent socket to http::CheckEof.  We can no longer
		    # treat bytes received as junk. The server might still time
		    # out and half-close the socket if it has not yet received
		    # the first "puts".
		    fileevent $sock readable {}
		}
	    }
	    puts -nonewline $sock \
		[string range $state(-query) $state(queryoffset) \
		     [expr {$state(queryoffset) + $state(-queryblocksize) - 1}]]
	    incr state(queryoffset) $state(-queryblocksize)
	    if {$state(queryoffset) >= $state(querylength)} {
		set state(queryoffset) $state(querylength)
		set done 1
	    }
	} else {
	    # Copy blocks from the query channel

	    set outStr [read $state(-querychannel) $state(-queryblocksize)]
	    if {[eof $state(-querychannel)]} {
		# This will be the last puts for the request-body.
		if {    (![catch {fileevent $sock readable} binding])
		     && ($binding eq [list http::CheckEof $sock])
		} {
		    # Remove the "fileevent readable" binding of an idle
		    # persistent socket to http::CheckEof.  We can no longer
		    # treat bytes received as junk. The server might still time
		    # out and half-close the socket if it has not yet received
		    # the first "puts".
		    fileevent $sock readable {}
		}
	    }
	    puts -nonewline $sock $outStr
	    incr state(queryoffset) [string length $outStr]
	    if {[eof $state(-querychannel)]} {
		set done 1
	    }
	}
    } err opts]} {
	# Do not call Finish here, but instead let the read half of the socket
	# process whatever server reply there is to get.
	set state(posterror) $err
	set info [dict get $opts -errorinfo]
	set code [dict get $opts -code]
	set state(postErrorFull) [list $err $info $code]
	set done 1
    }

    if {$done} {
	catch {flush $sock}
	fileevent $sock writable {}
	Log ^C$tk end sending request - token $token
	# End of writing (POST method).  The request has been sent.

	DoneRequest $token
    }

    # Callback to the client after we've completely handled everything.

    if {[string length $state(-queryprogress)]} {
	namespace eval :: $state(-queryprogress) \
	    [list $token $state(querylength) $state(queryoffset)]
    }
    return
}

# http::Event
#
#	Handle input on the socket. This command is the core of
#	the coroutine commands ${token}--EventCoroutine that are
#	bound to "fileevent $sock readable" and process input.
#
# Arguments
#	sock	The socket receiving input.
#	token	The token returned from http::geturl
#
# Side Effects
#	Read the socket and handle callbacks.

proc http::Event {sock token} {
    variable http
    variable socketMapping
    variable socketRdState
    variable socketWrState
    variable socketRdQueue
    variable socketWrQueue
    variable socketPhQueue
    variable socketClosing
    variable socketPlayCmd
    variable socketCoEvent
    variable socketProxyId

    variable $token
    upvar 0 $token state
    set tk [namespace tail $token]
    while 1 {
	yield
	##Log Event call - token $token

	if {![info exists state]} {
	    Log "Event $sock with invalid token '$token' - remote close?"
	    if {!([catch {eof $sock} tmp] || $tmp)} {
		if {[set d [read $sock]] ne ""} {
		    Log "WARNING: additional data left on closed socket\
			    - token $token"
		}
	    }
	    Log ^X$tk end of response (token error) - token $token
	    CloseSocket $sock
	    return
	}
	if {$state(state) eq "connecting"} {
	    ##Log - connecting - token $token
	    if {    $state(reusing)
		 && $state(-pipeline)
		 && ($state(-timeout) > 0)
		 && (![info exists state(after)])
	    } {
		set state(after) [after $state(-timeout) \
			[list http::reset $token timeout]]
	    }

	    if {[catch {gets $sock state(http)} nsl]} {
		Log "WARNING - if testing, pay special attention to this\
			case (GK) which is seldom executed - token $token"
		if {[info exists state(reusing)] && $state(reusing)} {
		    # The socket was closed at the server end, and closed at
		    # this end by http::CheckEof.

		    if {[TestForReplay $token read $nsl c]} {
			return
		    }
		    # else:
		    # This is NOT a persistent socket that has been closed since
		    # its last use.
		    # If any other requests are in flight or pipelined/queued,
		    # they will be discarded.
		} else {
		    # https handshake errors come here, for
		    # Tcl 9.0 with http::SecureProxyConnect.
		    set msg [registerError $sock]
		    registerError $sock {}
		    if {$msg eq {}} {
			set msg $nsl
		    }
		    Log ^X$tk end of response (error) - token $token
		    Finish $token $msg
		    return
		}
	    } elseif {$nsl >= 0} {
		##Log - connecting 1 - token $token
		set state(state) "header"
	    } elseif {    ([catch {eof $sock} tmp] || $tmp)
		       && [info exists state(reusing)]
		       && $state(reusing)
	    } {
		# The socket was closed at the server end, and we didn't notice.
		# This is the first read - where the closure is usually first
		# detected.

		if {[TestForReplay $token read {} d]} {
		    return
		}

		# else:
		# This is NOT a persistent socket that has been closed since its
		# last use.
		# If any other requests are in flight or pipelined/queued, they
		# will be discarded.
	    }
	} elseif {$state(state) eq "header"} {
	    if {[catch {gets $sock line} nhl]} {
		##Log header failed - token $token
		Log ^X$tk end of response (error) - token $token
		Finish $token $nhl
		return
	    } elseif {$nhl == 0} {
		##Log header done - token $token
		Log ^E$tk end of response headers - token $token
		# We have now read all headers
		# We ignore HTTP/1.1 100 Continue returns. RFC2616 sec 8.2.3
		if {    ($state(http) eq "")
		     || ([regexp {^\S+\s(\d+)} $state(http) {} x] && $x == 100)
		} {
		    set state(state) "connecting"
		    continue
		    # This was a "return" in the pre-coroutine code.
		}

		# We have $state(http) so let's split it into its components.
		if {[regexp {^HTTP/(\S+) ([0-9]{3}) (.*)$} $state(http) \
			-> httpResponse responseCode reasonPhrase]
		} {
		    set state(httpResponse) $httpResponse
		    set state(responseCode) $responseCode
		    set state(reasonPhrase) $reasonPhrase
		} else {
		    set state(httpResponse) $state(http)
		    set state(responseCode) $state(http)
		    set state(reasonPhrase) $state(http)
		}

		if {    ([info exists state(connection)])
		     && ([info exists socketMapping($state(socketinfo))])
		     && ("keep-alive" in $state(connection))
		     && ($state(-keepalive))
		     && (!$state(reusing))
		     && ($state(-pipeline))
		} {
		    # Response headers received for first request on a
		    # persistent socket.  Now ready for pipelined writes (if
		    # any).
		    # Previous value is $token. It cannot be "pending".
		    set socketWrState($state(socketinfo)) Wready
		    http::NextPipelinedWrite $token
		}

		# Once a "close" has been signaled, the client MUST NOT send any
		# more requests on that connection.
		#
		# If either the client or the server sends the "close" token in
		# the Connection header, that request becomes the last one for
		# the connection.

		if {    ([info exists state(connection)])
		     && ([info exists socketMapping($state(socketinfo))])
		     && ("close" in $state(connection))
		     && ($state(-keepalive))
		} {
		    # The server warns that it will close the socket after this
		    # response.
		    ##Log WARNING - socket will close after response for $token
		    # Prepare data for a call to ReplayIfClose.
		    Log $token socket will close after this transaction
		    # 1. Cancel socket-assignment coro events that have not yet
		    # launched, and add the tokens to the write queue.
		    if {[info exists socketCoEvent($state(socketinfo))]} {
			foreach {tok can} $socketCoEvent($state(socketinfo)) {
			    lappend socketWrQueue($state(socketinfo)) $tok
			    unset -nocomplain ${tok}(socketcoro)
			    after cancel $can
			    Log $tok Cancel socket after-idle event (Event)
			    Log Move $tok from socketCoEvent to socketWrQueue and cancel its after idle coro
			}
			set socketCoEvent($state(socketinfo)) {}
		    }

		    if {    ($socketRdQueue($state(socketinfo)) ne {})
			 || ($socketWrQueue($state(socketinfo)) ne {})
			 || ($socketWrState($state(socketinfo)) ni
						[list Wready peNding $token])
		    } {
			set InFlightW $socketWrState($state(socketinfo))
			if {$InFlightW in [list Wready peNding $token]} {
			    set InFlightW Wready
			} else {
			    set msg "token ${InFlightW} is InFlightW"
			    ##Log $msg - token $token
			}
			set socketPlayCmd($state(socketinfo)) \
				[list ReplayIfClose $InFlightW \
				$socketRdQueue($state(socketinfo)) \
				$socketWrQueue($state(socketinfo))]

			# - All tokens are preserved for re-use by ReplayCore.
			# - Queues are preserved in case of Finish with error,
			#   but are not used for anything else because
			#   socketClosing(*) is set below.
			# - Cancel the state(after) timeout events.
			foreach tokenVal $socketRdQueue($state(socketinfo)) {
			    if {[info exists ${tokenVal}(after)]} {
				after cancel [set ${tokenVal}(after)]
				unset ${tokenVal}(after)
			    }
			    # Tokens in the read queue have no (socketcoro) to
			    # cancel.
			}
		    } else {
			set socketPlayCmd($state(socketinfo)) \
				{ReplayIfClose Wready {} {}}
		    }

		    # Do not allow further connections on this socket (but
		    # geturl can add new requests to the replay).
		    set socketClosing($state(socketinfo)) 1
		}

		set state(state) body

		# According to
		# https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Connection
		# any comma-separated "Connection:" list implies keep-alive, but I
		# don't see this in the RFC so we'll play safe and
		# scan any list for "close".
		# Done here to support combining duplicate header field's values.
		if {   [info exists state(connection)]
		    && ("close" ni $state(connection))
		    && ("keep-alive" ni $state(connection))
		} {
		    lappend state(connection) "keep-alive"
		}

		# If doing a HEAD, then we won't get any body
		if {$state(-validate)} {
		    Log ^F$tk end of response for HEAD request - token $token
		    set state(state) complete
		    Eot $token
		    return
		} elseif {
			($state(method) eq {CONNECT})
		     && [string is integer -strict $state(responseCode)]
		     && ($state(responseCode) >= 200)
		     && ($state(responseCode) < 300)
		} {
		    # A successful CONNECT response has no body.
		    # (An unsuccessful CONNECT has headers and body.)
		    # The code below is abstracted from Eot/Finish, but
		    # keeps the socket open.
		    catch {fileevent $state(sock) readable {}}
		    catch {fileevent $state(sock) writable {}}
		    set state(state) complete
		    set state(status) ok
		    if {[info commands ${token}--EventCoroutine] ne {}} {
			rename ${token}--EventCoroutine {}
		    }
		    if {[info commands ${token}--SocketCoroutine] ne {}} {
			rename ${token}--SocketCoroutine {}
		    }
		    if {[info exists state(socketcoro)]} {
			Log $token Cancel socket after-idle event (Finish)
			after cancel $state(socketcoro)
			unset state(socketcoro)
		    }
		    if {[info exists state(after)]} {
			after cancel $state(after)
			unset state(after)
		    }
		    if {    [info exists state(-command)]
			 && (![info exists state(done-command-cb)])
		    } {
			set state(done-command-cb) yes
			if {[catch {namespace eval :: $state(-command) $token} err]} {
			    set state(error) [list $err $errorInfo $errorCode]
			    set state(status) error
			}
		    }
		    return
		}

		# - For non-chunked transfer we may have no body - in this case
		#   we may get no further file event if the connection doesn't
		#   close and no more data is sent. We can tell and must finish
		#   up now - not later - the alternative would be to wait until
		#   the server times out.
		# - In this case, the server has NOT told the client it will
		#   close the connection, AND it has NOT indicated the resource
		#   length EITHER by setting the Content-Length (totalsize) OR
		#   by using chunked Transfer-Encoding.
		# - Do not worry here about the case (Connection: close) because
		#   the server should close the connection.
		# - IF (NOT Connection: close) AND (NOT chunked encoding) AND
		#      (totalsize == 0).

		if {    (!(    [info exists state(connection)]
			    && ("close" in $state(connection))
			  )
			)
		     && ($state(transfer) eq {})
		     && ($state(totalsize) == 0)
		} {
		    set msg {body size is 0 and no events likely - complete}
		    Log "$msg - token $token"
		    set msg {(length unknown, set to 0)}
		    Log ^F$tk end of response body {*}$msg - token $token
		    set state(state) complete
		    Eot $token
		    return
		}

		# We have to use binary translation to count bytes properly.
		lassign [fconfigure $sock -translation] trRead trWrite
		fconfigure $sock -translation [list binary $trWrite]

		if {
		    $state(-binary) || [IsBinaryContentType $state(type)]
		} {
		    # Turn off conversions for non-text data.
		    set state(binary) 1
		}
		if {[info exists state(-channel)]} {
		    if {$state(binary) || [llength [ContentEncoding $token]]} {
			fconfigure $state(-channel) -translation binary
		    }
		    if {![info exists state(-handler)]} {
			# Initiate a sequence of background fcopies.
			fileevent $sock readable {}
			rename ${token}--EventCoroutine {}
			CopyStart $sock $token
			return
		    }
		}
	    } elseif {$nhl > 0} {
		# Process header lines.
		##Log header - token $token - $line
		if {[regexp -nocase {^([^:]+):(.+)$} $line x key value]} {
		    set key [string tolower $key]
		    switch -- $key {
			content-type {
			    set state(type) [string trim [string tolower $value]]
			    # Grab the optional charset information.
			    if {[regexp -nocase \
				    {charset\s*=\s*\"((?:[^""]|\\\")*)\"} \
				    $state(type) -> cs]} {
				set state(charset) [string map {{\"} \"} $cs]
			    } else {
				regexp -nocase {charset\s*=\s*(\S+?);?} \
					$state(type) -> state(charset)
			    }
			}
			content-length {
			    set state(totalsize) [string trim $value]
			}
			content-encoding {
			    set state(coding) [string trim $value]
			}
			transfer-encoding {
			    set state(transfer) \
				    [string trim [string tolower $value]]
			}
			proxy-connection -
			connection {
			    # RFC 7230 Section 6.1 states that a comma-separated
			    # list is an acceptable value.
			    if {![info exists state(connectionRespFlag)]} {
				# This is the first "Connection" response header.
				# Scrub the earlier value set by iniitialisation.
				set state(connectionRespFlag) {}
				set state(connection) {}
			    }
			    foreach el [SplitCommaSeparatedFieldValue $value] {
				lappend state(connection) [string tolower $el]
			    }
			}
			upgrade {
			    set state(upgrade) [string trim $value]
			}
			set-cookie {
			    if {$http(-cookiejar) ne ""} {
				ParseCookie $token [string trim $value]
			    }
			}
		    }
		    lappend state(meta) $key [string trim $value]
		}
	    }
	} else {
	    # Now reading body
	    ##Log body - token $token
	    if {[catch {
		if {[info exists state(-handler)]} {
		    set n [namespace eval :: $state(-handler) [list $sock $token]]
		    ##Log handler $n - token $token
		    # N.B. the protocol has been set to 1.0 because the -handler
		    # logic is not expected to handle chunked encoding.
		    # FIXME Allow -handler with 1.1 on dechunked stacked chan.
		    if {$state(totalsize) == 0} {
			# We know the transfer is complete only when the server
			# closes the connection - i.e. eof is not an error.
			set state(state) complete
		    }
		    if {![string is integer -strict $n]} {
			if 1 {
			    # Do not tolerate bad -handler - fail with error
			    # status.
			    set msg {the -handler command for http::geturl must\
				    return an integer (the number of bytes\
				    read)}
			    Log ^X$tk end of response (handler error) -\
				    token $token
			    Eot $token $msg
			} else {
			    # Tolerate the bad -handler, and continue.  The
			    # penalty:
			    # (a) Because the handler returns nonsense, we know
			    #     the transfer is complete only when the server
			    #     closes the connection - i.e. eof is not an
			    #     error.
			    # (b) http::size will not be accurate.
			    # (c) The transaction is already downgraded to 1.0
			    #     to avoid chunked transfer encoding.  It MUST
			    #     also be forced to "Connection: close" or the
			    #     HTTP/1.0 equivalent; or it MUST fail (as
			    #     above) if the server sends
			    #     "Connection: keep-alive" or the HTTP/1.0
			    #     equivalent.
			    set n 0
			    set state(state) complete
			}
		    }
		} elseif {[info exists state(transfer_final)]} {
		    # This code forgives EOF in place of the final CRLF.
		    set line [GetTextLine $sock]
		    set n [string length $line]
		    set state(state) complete
		    if {$n > 0} {
			# - HTTP trailers (late response headers) are permitted
			#   by Chunked Transfer-Encoding, and can be safely
			#   ignored.
			# - Do not count these bytes in the total received for
			#   the response body.
			Log "trailer of $n bytes after final chunk -\
				token $token"
			append state(transfer_final) $line
			set n 0
		    } else {
			Log ^F$tk end of response body (chunked) - token $token
			Log "final chunk part - token $token"
			Eot $token
		    }
		} elseif {    [info exists state(transfer)]
			   && ($state(transfer) eq "chunked")
		} {
		    ##Log chunked - token $token
		    set size 0
		    set hexLenChunk [GetTextLine $sock]
		    #set ntl [string length $hexLenChunk]
		    if {[string trim $hexLenChunk] ne ""} {
			scan $hexLenChunk %x size
			if {$size != 0} {
			    ##Log chunk-measure $size - token $token
			    set chunk [BlockingRead $sock $size]
			    set n [string length $chunk]
			    if {$n >= 0} {
				append state(body) $chunk
				incr state(log_size) [string length $chunk]
				##Log chunk $n cumul $state(log_size) -\
					token $token
			    }
			    if {$size != [string length $chunk]} {
				Log "WARNING: mis-sized chunk:\
				    was [string length $chunk], should be\
				    $size - token $token"
				set n 0
				set state(connection) close
				Log ^X$tk end of response (chunk error) \
					- token $token
				set msg {error in chunked encoding - fetch\
					terminated}
				Eot $token $msg
			    }
			    # CRLF that follows chunk.
			    # If eof, this is handled at the end of this proc.
			    GetTextLine $sock
			} else {
			    set n 0
			    set state(transfer_final) {}
			}
		    } else {
			# Line expected to hold chunk length is empty, or eof.
			##Log bad-chunk-measure - token $token
			set n 0
			set state(connection) close
			Log ^X$tk end of response (chunk error) - token $token
			Eot $token {error in chunked encoding -\
				fetch terminated}
		    }
		} else {
		    ##Log unchunked - token $token
		    if {$state(totalsize) == 0} {
			# We know the transfer is complete only when the server
			# closes the connection.
			set state(state) complete
			set reqSize $state(-blocksize)
		    } else {
			# Ask for the whole of the unserved response-body.
			# This works around a problem with a tls::socket - for
			# https in keep-alive mode, and a request for
			# $state(-blocksize) bytes, the last part of the
			# resource does not get read until the server times out.
			set reqSize [expr {  $state(totalsize)
					   - $state(currentsize)}]

			# The workaround fails if reqSize is
			# capped at $state(-blocksize).
			# set reqSize [expr {min($reqSize, $state(-blocksize))}]
		    }
		    set c $state(currentsize)
		    set t $state(totalsize)
		    ##Log non-chunk currentsize $c of totalsize $t -\
			    token $token
		    set block [read $sock $reqSize]
		    set n [string length $block]
		    if {$n >= 0} {
			append state(body) $block
			##Log non-chunk [string length $state(body)] -\
				token $token
		    }
		}
		# This calculation uses n from the -handler, chunked, or
		# unchunked case as appropriate.
		if {[info exists state]} {
		    if {$n >= 0} {
			incr state(currentsize) $n
			set c $state(currentsize)
			set t $state(totalsize)
			##Log another $n currentsize $c totalsize $t -\
				token $token
		    }
		    # If Content-Length - check for end of data.
		    if {
			   ($state(totalsize) > 0)
			&& ($state(currentsize) >= $state(totalsize))
		    } {
			Log ^F$tk end of response body (unchunked) -\
				token $token
			set state(state) complete
			Eot $token
		    }
		}
	    } err]} {
		Log ^X$tk end of response (error ${err}) - token $token
		Finish $token $err
		return
	    } else {
		if {[info exists state(-progress)]} {
		    namespace eval :: $state(-progress) \
			[list $token $state(totalsize) $state(currentsize)]
		}
	    }
	}

	# catch as an Eot above may have closed the socket already
	# $state(state) may be connecting, header, body, or complete
	if {(![catch {eof $sock} eof]) && $eof} {
	    # [eof sock] succeeded and the result was 1
	    ##Log eof - token $token
	    if {[info exists $token]} {
		set state(connection) close
		if {$state(state) eq "complete"} {
		    # This includes all cases in which the transaction
		    # can be completed by eof.
		    # The value "complete" is set only in http::Event, and it is
		    # used only in the test above.
		    Log ^F$tk end of response body (unchunked, eof) -\
			    token $token
		    Eot $token
		} else {
		    # Premature eof.
		    Log ^X$tk end of response (unexpected eof) - token $token
		    Eot $token eof
		}
	    } else {
		# open connection closed on a token that has been cleaned up.
		Log ^X$tk end of response (token error) - token $token
		CloseSocket $sock
	    }
	} else {
	    # EITHER [eof sock] failed - presumed done by Eot
	    # OR     [eof sock] succeeded and the result was 0
	}
    }
    return
}

# http::TestForReplay
#
#	Command called if eof is discovered when a socket is first used for a
#	new transaction.  Typically this occurs if a persistent socket is used
#	after a period of idleness and the server has half-closed the socket.
#
# token  - the connection token returned by http::geturl
# doing  - "read" or "write"
# err    - error message, if any
# caller - code to identify the caller - used only in logging
#
# Return Value: boolean, true iff the command calls http::ReplayIfDead.

proc http::TestForReplay {token doing err caller} {
    variable http
    variable $token
    upvar 0 $token state
    set tk [namespace tail $token]
    if {$doing eq "read"} {
	set code Q
	set action response
	set ing reading
    } else {
	set code P
	set action request
	set ing writing
    }

    if {$err eq {}} {
	set err "detect eof when $ing (server timed out?)"
    }

    if {$state(method) eq "POST" && !$http(-repost)} {
	# No Replay.
	# The present transaction will end when Finish is called.
	# That call to Finish will abort any other transactions
	# currently in the write queue.
	# For calls from http::Event this occurs when execution
	# reaches the code block at the end of that proc.
	set msg {no retry for POST with http::config -repost 0}
	Log reusing socket failed "($caller)" - $msg - token $token
	Log error - $err - token $token
	Log ^X$tk end of $action (error) - token $token
	return 0
    } else {
	# Replay.
	set msg {try a new socket}
	Log reusing socket failed "($caller)" - $msg - token $token
	Log error - $err - token $token
	Log ^$code$tk Any unfinished (incl this one) failed - token $token
	ReplayIfDead $token $doing
	return 1
    }
}

# http::IsBinaryContentType --
#
#	Determine if the content-type means that we should definitely transfer
#	the data as binary. [Bug 838e99a76d]
#
# Arguments
#	type	The content-type of the data.
#
# Results:
#	Boolean, true if we definitely should be binary.

proc http::IsBinaryContentType {type} {
    lassign [split [string tolower $type] "/;"] major minor
    if {$major eq "text"} {
	return false
    }
    # There's a bunch of XML-as-application-format things about. See RFC 3023
    # and so on.
    if {$major eq "application"} {
	set minor [string trimright $minor]
	if {$minor in {"json" "xml" "xml-external-parsed-entity" "xml-dtd"}} {
	    return false
	}
    }
    # Not just application/foobar+xml but also image/svg+xml, so let us not
    # restrict things for now...
    if {[string match "*+xml" $minor]} {
	return false
    }
    return true
}

proc http::ParseCookie {token value} {
    variable http
    variable CookieRE
    variable $token
    upvar 0 $token state

    if {![regexp $CookieRE $value -> cookiename cookieval opts]} {
	# Bad cookie! No biscuit!
	return
    }

    # Convert the options into a list before feeding into the cookie store;
    # ugly, but quite easy.
    set realopts {hostonly 1 path / secure 0 httponly 0}
    dict set realopts origin $state(host)
    dict set realopts domain $state(host)
    foreach option [split [regsub -all {;\s+} $opts \u0000] \u0000] {
	regexp {^(.*?)(?:=(.*))?$} $option -> optname optval
	switch -exact -- [string tolower $optname] {
	    expires {
		if {[catch {
		    #Sun, 06 Nov 1994 08:49:37 GMT
		    dict set realopts expires \
			[clock scan $optval -format "%a, %d %b %Y %T %Z"]
		}] && [catch {
		    # Google does this one
		    #Mon, 01-Jan-1990 00:00:00 GMT
		    dict set realopts expires \
			[clock scan $optval -format "%a, %d-%b-%Y %T %Z"]
		}] && [catch {
		    # This is in the RFC, but it is also in the original
		    # Netscape cookie spec, now online at:
		    # <URL:http://curl.haxx.se/rfc/cookie_spec.html>
		    #Sunday, 06-Nov-94 08:49:37 GMT
		    dict set realopts expires \
			[clock scan $optval -format "%A, %d-%b-%y %T %Z"]
		}]} {catch {
		    #Sun Nov  6 08:49:37 1994
		    dict set realopts expires \
			[clock scan $optval -gmt 1 -format "%a %b %d %T %Y"]
		}}
	    }
	    max-age {
		# Normalize
		if {[string is integer -strict $optval]} {
		    dict set realopts expires [expr {[clock seconds] + $optval}]
		}
	    }
	    domain {
		# From the domain-matches definition [RFC 2109, section 2]:
		#   Host A's name domain-matches host B's if [...]
		#	A is a FQDN string and has the form NB, where N is a
		#	non-empty name string, B has the form .B', and B' is a
		#	FQDN string. (So, x.y.com domain-matches .y.com but
		#	not y.com.)
		if {$optval ne "" && ![string match *. $optval]} {
		    dict set realopts domain [string trimleft $optval "."]
		    dict set realopts hostonly [expr {
			! [string match .* $optval]
		    }]
		}
	    }
	    path {
		if {[string match /* $optval]} {
		    dict set realopts path $optval
		}
	    }
	    secure - httponly {
		dict set realopts [string tolower $optname] 1
	    }
	}
    }
    dict set realopts key $cookiename
    dict set realopts value $cookieval
    {*}$http(-cookiejar) storeCookie $realopts
}

# http::GetTextLine --
#
#	Get one line with the stream in crlf mode.
#	Used if Transfer-Encoding is chunked, to read the line that
#	reports the size of the following chunk.
#	Empty line is not distinguished from eof.  The caller must
#	be able to handle this.
#
# Arguments
#	sock	The socket receiving input.
#
# Results:
#	The line of text, without trailing newline

proc http::GetTextLine {sock} {
    set tr [fconfigure $sock -translation]
    lassign $tr trRead trWrite
    fconfigure $sock -translation [list crlf $trWrite]
    set r [BlockingGets $sock]
    fconfigure $sock -translation $tr
    return $r
}

# http::BlockingRead
#
#	Replacement for a blocking read.
#	The caller must be a coroutine.
#	Used when we expect to read a chunked-encoding
#	chunk of known size.

proc http::BlockingRead {sock size} {
    if {$size < 1} {
	return
    }
    set result {}
    while 1 {
	set need [expr {$size - [string length $result]}]
	set block [read $sock $need]
	set eof [expr {[catch {eof $sock} tmp] || $tmp}]
	append result $block
	if {[string length $result] >= $size || $eof} {
	    return $result
	} else {
	    yield
	}
    }
}

# http::BlockingGets
#
#	Replacement for a blocking gets.
#	The caller must be a coroutine.
#	Empty line is not distinguished from eof.  The caller must
#	be able to handle this.

proc http::BlockingGets {sock} {
    while 1 {
	set count [gets $sock line]
	set eof [expr {[catch {eof $sock} tmp] || $tmp}]
	if {$count >= 0 || $eof} {
	    return $line
	} else {
	    yield
	}
    }
}

# http::CopyStart
#
#	Error handling wrapper around fcopy
#
# Arguments
#	sock	The socket to copy from
#	token	The token returned from http::geturl
#
# Side Effects
#	This closes the connection upon error

proc http::CopyStart {sock token {initial 1}} {
    upvar 0 $token state
    if {[info exists state(transfer)] && $state(transfer) eq "chunked"} {
	foreach coding [ContentEncoding $token] {
	    if {$coding eq {deflateX}} {
		# Use the standards-compliant choice.
		set coding2 decompress
	    } else {
		set coding2 $coding
	    }
	    lappend state(zlib) [zlib stream $coding2]
	}
	MakeTransformationChunked $sock [namespace code [list CopyChunk $token]]
    } else {
	if {$initial} {
	    foreach coding [ContentEncoding $token] {
		if {$coding eq {deflateX}} {
		    # Use the standards-compliant choice.
		    set coding2 decompress
		} else {
		    set coding2 $coding
		}
		zlib push $coding2 $sock
	    }
	}
	if {[catch {
	    # FIXME Keep-Alive on https tls::socket with unchunked transfer
	    # hangs until the server times out. A workaround is possible, as for
	    # the case without -channel, but it does not use the neat "fcopy"
	    # solution.
	    fcopy $sock $state(-channel) -size $state(-blocksize) -command \
		[list http::CopyDone $token]
	} err]} {
	    Finish $token $err
	}
    }
    return
}

proc http::CopyChunk {token chunk} {
    upvar 0 $token state
    if {[set count [string length $chunk]]} {
	incr state(currentsize) $count
	if {[info exists state(zlib)]} {
	    foreach stream $state(zlib) {
		set chunk [$stream add $chunk]
	    }
	}
	puts -nonewline $state(-channel) $chunk
	if {[info exists state(-progress)]} {
	    namespace eval :: [linsert $state(-progress) end \
		      $token $state(totalsize) $state(currentsize)]
	}
    } else {
	Log "CopyChunk Finish - token $token"
	if {[info exists state(zlib)]} {
	    set excess ""
	    foreach stream $state(zlib) {
		catch {
		    $stream put -finalize $excess
		    set excess ""
		    set overflood ""
		    while {[set overflood [$stream get]] ne ""} { append excess $overflood }
		}
	    }
	    puts -nonewline $state(-channel) $excess
	    foreach stream $state(zlib) { $stream close }
	    unset state(zlib)
	}
	Eot $token ;# FIX ME: pipelining.
    }
    return
}

# http::CopyDone
#
#	fcopy completion callback
#
# Arguments
#	token	The token returned from http::geturl
#	count	The amount transferred
#
# Side Effects
#	Invokes callbacks

proc http::CopyDone {token count {error {}}} {
    variable $token
    upvar 0 $token state
    set sock $state(sock)
    incr state(currentsize) $count
    if {[info exists state(-progress)]} {
	namespace eval :: $state(-progress) \
	    [list $token $state(totalsize) $state(currentsize)]
    }
    # At this point the token may have been reset.
    if {[string length $error]} {
	Finish $token $error
    } elseif {[catch {eof $sock} iseof] || $iseof} {
	Eot $token
    } else {
	CopyStart $sock $token 0
    }
    return
}

# http::Eot
#
#	Called when either:
#	a. An eof condition is detected on the socket.
#	b. The client decides that the response is complete.
#	c. The client detects an inconsistency and aborts the transaction.
#
#	Does:
#	1. Set state(status)
#	2. Reverse any Content-Encoding
#	3. Convert charset encoding and line ends if necessary
#	4. Call http::Finish
#
# Arguments
#	token	The token returned from http::geturl
#	force	(previously) optional, has no effect
#	reason	- "eof" means premature EOF (not EOF as the natural end of
#		  the response)
#		- "" means completion of response, with or without EOF
#		- anything else describes an error condition other than
#		  premature EOF.
#
# Side Effects
#	Clean up the socket

proc http::Eot {token {reason {}}} {
    variable $token
    upvar 0 $token state
    if {$reason eq "eof"} {
	# Premature eof.
	set state(status) eof
	set reason {}
    } elseif {$reason ne ""} {
	# Abort the transaction.
	set state(status) $reason
    } else {
	# The response is complete.
	set state(status) ok
    }

    if {[string length $state(body)] > 0} {
	if {[catch {
	    foreach coding [ContentEncoding $token] {
		if {$coding eq {deflateX}} {
		    # First try the standards-compliant choice.
		    set coding2 decompress
		    if {[catch {zlib $coding2 $state(body)} result]} {
			# If that fails, try the MS non-compliant choice.
			set coding2 inflate
			set state(body) [zlib $coding2 $state(body)]
		    } else {
			# error {failed at standards-compliant deflate}
			set state(body) $result
		    }
		} else {
		    set state(body) [zlib $coding $state(body)]
		}
	    }
	} err]} {
	    Log "error doing decompression for token $token: $err"
	    Finish $token $err
	    return
	}

	if {!$state(binary)} {
	    # If we are getting text, set the incoming channel's encoding
	    # correctly.  iso8859-1 is the RFC default, but this could be any
	    # IANA charset.  However, we only know how to convert what we have
	    # encodings for.

	    set enc [CharsetToEncoding $state(charset)]
	    if {$enc ne "binary"} {
		if {[package vsatisfies [package provide Tcl] 9.0-]} {
		    set state(body) [encoding convertfrom -profile replace $enc $state(body)]
		} else {
		    set state(body) [encoding convertfrom $enc $state(body)]
		}
	    }

	    # Translate text line endings.
	    set state(body) [string map {\r\n \n \r \n} $state(body)]
	}
	if {[info exists state(-guesstype)] && $state(-guesstype)} {
	    GuessType $token
	}
    }
    Finish $token $reason
    return
}


# ------------------------------------------------------------------------------
#  Proc http::GuessType
# ------------------------------------------------------------------------------
# Command to attempt limited analysis of a resource with undetermined
# Content-Type, i.e. "application/octet-stream".  This value can be set for two
# reasons:
# (a) by the server, in a Content-Type header
# (b) by http::geturl, as the default value if the server does not supply a
#     Content-Type header.
#
# This command converts a resource if:
# (1) it has type application/octet-stream
# (2) it begins with an XML declaration "<?xml name="value" ... >?"
# (3) one tag is named "encoding" and has a recognised value; or no "encoding"
#     tag exists (defaulting to utf-8)
#
# RFC 9110 Sec. 8.3 states:
# "If a Content-Type header field is not present, the recipient MAY either
# assume a media type of "application/octet-stream" ([RFC2046], Section 4.5.1)
# or examine the data to determine its type."
#
# The RFC goes on to describe the pitfalls of "MIME sniffing", including
# possible security risks.
#
# Arguments:
# token       - connection token
#
# Return Value: (boolean) true iff a change has been made
# ------------------------------------------------------------------------------

proc http::GuessType {token} {
    variable $token
    upvar 0 $token state

    if {$state(type) ne {application/octet-stream}} {
	return 0
    }

    set body $state(body)
    # e.g. {<?xml version="1.0" encoding="utf-8"?> ...}

    if {![regexp -nocase -- {^<[?]xml[[:space:]][^>?]*[?]>} $body match]} {
	return 0
    }
    # e.g. {<?xml version="1.0" encoding="utf-8"?>}

    set contents [regsub -- {[[:space:]]+} $match { }]
    set contents [string range [string tolower $contents] 6 end-2]
    # e.g. {version="1.0" encoding="utf-8"}
    # without excess whitespace or upper-case letters

    if {![regexp -- {^([^=" ]+="[^"]+" )+$} "$contents "]} {
	return 0
    }
    # The application/xml default encoding:
    set res utf-8

    set tagList [regexp -all -inline -- {[^=" ]+="[^"]+"} $contents]
    foreach tag $tagList {
	regexp -- {([^=" ]+)="([^"]+)"} $tag -> name value
	if {$name eq {encoding}} {
	    set res $value
	}
    }
    set enc [CharsetToEncoding $res]
    if {$enc eq "binary"} {
	return 0
    }
    if {[package vsatisfies [package provide Tcl] 9.0-]} {
	set state(body) [encoding convertfrom -profile replace $enc $state(body)]
    } else {
	set state(body) [encoding convertfrom $enc $state(body)]
    }
    set state(body) [string map {\r\n \n \r \n} $state(body)]
    set state(type) application/xml
    set state(binary) 0
    set state(charset) $res
    return 1
}


# http::wait --
#
#	See documentation for details.
#
# Arguments:
#	token	Connection token.
#
# Results:
#	The status after the wait.

proc http::wait {token} {
    variable $token
    upvar 0 $token state

    if {![info exists state(status)] || $state(status) eq ""} {
	# We must wait on the original variable name, not the upvar alias
	vwait ${token}(status)
    }

    return [status $token]
}

# http::formatQuery --
#
#	See documentation for details.  Call http::formatQuery with an even
#	number of arguments, where the first is a name, the second is a value,
#	the third is another name, and so on.
#
# Arguments:
#	args	A list of name-value pairs.
#
# Results:
#	TODO

proc http::formatQuery {args} {
    if {[llength $args] % 2} {
	return \
	    -code error \
	    -errorcode [list HTTP BADARGCNT $args] \
	    {Incorrect number of arguments, must be an even number.}
    }
    set result ""
    set sep ""
    foreach i $args {
	append result $sep [quoteString $i]
	if {$sep eq "="} {
	    set sep &
	} else {
	    set sep =
	}
    }
    return $result
}

# http::quoteString --
#
#	Do x-www-urlencoded character mapping
#
# Arguments:
#	string	The string the needs to be encoded
#
# Results:
#       The encoded string

proc http::quoteString {string} {
    variable http
    variable formMap

    # The spec says: "non-alphanumeric characters are replaced by '%HH'". Use
    # a pre-computed map and [string map] to do the conversion (much faster
    # than [regsub]/[subst]). [Bug 1020491]

    if {[package vsatisfies [package provide Tcl] 9.0-]} {
	set string [encoding convertto -profile replace $http(-urlencoding) $string]
    } else {
	set string [encoding convertto $http(-urlencoding) $string]
    }
    return [string map $formMap $string]
}

# http::ProxyRequired --
#	Default proxy filter.
#
# Arguments:
#	host	The destination host
#
# Results:
#       The current proxy settings

proc http::ProxyRequired {host} {
    variable http
    if {(![info exists http(-proxyhost)]) || ($http(-proxyhost) eq {})} {
	return
    }
    if {![info exists http(-proxyport)] || ($http(-proxyport) eq {})} {
	set port 8080
    } else {
	set port $http(-proxyport)
    }

    # Simple test (cf. autoproxy) for hosts that must be accessed directly,
    # not through the proxy server.
    foreach domain $http(-proxynot) {
	if {[string match -nocase $domain $host]} {
	    return {}
	}
    }
    return [list $http(-proxyhost) $port]
}

# http::CharsetToEncoding --
#
#	Tries to map a given IANA charset to a tcl encoding.  If no encoding
#	can be found, returns binary.
#

proc http::CharsetToEncoding {charset} {
    variable encodings

    set charset [string tolower $charset]
    if {[regexp {iso-?8859-([0-9]+)} $charset -> num]} {
	set encoding "iso8859-$num"
    } elseif {[regexp {iso-?2022-(jp|kr)} $charset -> ext]} {
	set encoding "iso2022-$ext"
    } elseif {[regexp {shift[-_]?jis} $charset]} {
	set encoding "shiftjis"
    } elseif {[regexp {(?:windows|cp)-?([0-9]+)} $charset -> num]} {
	set encoding "cp$num"
    } elseif {$charset eq "us-ascii"} {
	set encoding "ascii"
    } elseif {[regexp {(?:iso-?)?lat(?:in)?-?([0-9]+)} $charset -> num]} {
	switch -- $num {
	    5 {set encoding "iso8859-9"}
	    1 - 2 - 3 {
		set encoding "iso8859-$num"
	    }
	    default {
		set encoding "binary"
	    }
	}
    } else {
	# other charset, like euc-xx, utf-8,...  may directly map to encoding
	set encoding $charset
    }
    set idx [lsearch -exact $encodings $encoding]
    if {$idx >= 0} {
	return $encoding
    } else {
	return "binary"
    }
}


# ------------------------------------------------------------------------------
#  Proc http::ContentEncoding
# ------------------------------------------------------------------------------
# Return the list of content-encoding transformations we need to do in order.
#
    # --------------------------------------------------------------------------
    # Options for Accept-Encoding, Content-Encoding: the switch command
    # --------------------------------------------------------------------------
    # The symbol deflateX allows http to attempt both versions of "deflate",
    # unless there is a -channel - for a -channel, only "decompress" is tried.
    # Alternative/extra lines for switch:
    # The standards-compliant version of "deflate" can be chosen with:
    #		deflate { lappend r decompress }
    # The Microsoft non-compliant version of "deflate" can be chosen with:
    #		deflate { lappend r inflate }
    # The previously used implementation of "compress", which appears to be
    # incorrect and is rarely used by web servers, can be chosen with:
    #		compress - x-compress { lappend r decompress }
    # --------------------------------------------------------------------------
#
# Arguments:
# token  - Connection token.
#
# Return Value: list
# ------------------------------------------------------------------------------

proc http::ContentEncoding {token} {
    upvar 0 $token state
    set r {}
    if {[info exists state(coding)]} {
	foreach coding [split $state(coding) ,] {
	    switch -exact -- $coding {
		deflate { lappend r deflateX }
		gzip - x-gzip { lappend r gunzip }
		identity {}
		br {
		    return -code error\
			    "content-encoding \"br\" not implemented"
		}
		default {
		    Log "unknown content-encoding \"$coding\" ignored"
		}
	    }
	}
    }
    return $r
}

proc http::ReceiveChunked {chan command} {
    set data ""
    set size -1
    yield
    while {1} {
	chan configure $chan -translation {crlf binary}
	while {[gets $chan line] < 1} { yield }
	chan configure $chan -translation {binary binary}
	if {[scan $line %x size] != 1} {
	    return -code error "invalid size: \"$line\""
	}
	set chunk ""
	while {$size && ![chan eof $chan]} {
	    set part [chan read $chan $size]
	    incr size -[string length $part]
	    append chunk $part
	}
	if {[catch {
	    uplevel #0 [linsert $command end $chunk]
	}]} {
	    http::Log "Error in callback: $::errorInfo"
	}
	if {[string length $chunk] == 0} {
	    # channel might have been closed in the callback
	    catch {chan event $chan readable {}}
	    return
	}
    }
}

# http::SplitCommaSeparatedFieldValue --
#	Return the individual values of a comma-separated field value.
#
# Arguments:
#	fieldValue	Comma-separated header field value.
#
# Results:
#       List of values.
proc http::SplitCommaSeparatedFieldValue {fieldValue} {
    set r {}
    foreach el [split $fieldValue ,] {
	lappend r [string trim $el]
    }
    return $r
}


# http::GetFieldValue --
#	Return the value of a header field.
#
# Arguments:
#	headers	Headers key-value list
#	fieldName	Name of header field whose value to return.
#
# Results:
#       The value of the fieldName header field
#
# Field names are matched case-insensitively (RFC 7230 Section 3.2).
#
# If the field is present multiple times, it is assumed that the field is
# defined as a comma-separated list and the values are combined (by separating
# them with commas, see RFC 7230 Section 3.2.2) and returned at once.
proc http::GetFieldValue {headers fieldName} {
    set r {}
    foreach {field value} $headers {
	if {[string equal -nocase $fieldName $field]} {
	    if {$r eq {}} {
		set r $value
	    } else {
		append r ", $value"
	    }
	}
    }
    return $r
}

proc http::MakeTransformationChunked {chan command} {
    coroutine [namespace current]::dechunk$chan ::http::ReceiveChunked $chan $command
    chan event $chan readable [namespace current]::dechunk$chan
    return
}

interp alias {} http::data {} http::responseBody
interp alias {} http::code {} http::responseLine
interp alias {} http::mapReply {} http::quoteString
interp alias {} http::meta {} http::responseHeaders
interp alias {} http::metaValue {} http::responseHeaderValue
interp alias {} http::ncode {} http::responseCode


# ------------------------------------------------------------------------------
#  Proc http::socketAsCallback
# ------------------------------------------------------------------------------
# Command to use in place of ::socket as the value of ::tls::socketCmd.
# This command does the same as http::AltSocket, and also handles https
# connections through a proxy server.
#
# Notes.
# - The proxy server works differently for https and http.  This implementation
#   is for https.  The proxy for http is implemented in http::CreateToken (in
#   code that was previously part of http::geturl).
# - This code implicitly uses the tls options set for https in a call to
#   http::register, and does not need to call commands tls::*.  This simple
#   implementation is possible because tls uses a callback to ::socket that can
#   be redirected by changing the value of ::tls::socketCmd.
#
# Arguments:
# args        - as for ::socket
#
# Return Value: a socket identifier
# ------------------------------------------------------------------------------

proc http::socketAsCallback {args} {
    variable http

    set targ [lsearch -exact $args -type]
    if {$targ != -1} {
	set token [lindex $args $targ+1]
	upvar 0 ${token} state
	set protoProxyConn $state(protoProxyConn)
    } else {
	set protoProxyConn 0
    }

    set host [lindex $args end-1]
    set port [lindex $args end]
    if {    ($http(-proxyfilter) ne {})
	 && (![catch {$http(-proxyfilter) $host} proxy])
	 && $protoProxyConn
    } {
	set phost [lindex $proxy 0]
	set pport [lindex $proxy 1]
    } else {
	set phost {}
	set pport {}
    }
    if {$phost eq ""} {
	set sock [::http::AltSocket {*}$args]
    } else {
	set sock [::http::SecureProxyConnect {*}$args $phost $pport]
    }
    return $sock
}


# ------------------------------------------------------------------------------
#  Proc http::SecureProxyConnect
# ------------------------------------------------------------------------------
# Command to open a socket through a proxy server to a remote server for use by
# tls. The caller must perform the tls handshake.
#
# Notes
# - Based on patch supplied by Melissa Chawla in ticket 1173760, and
#   Proxy-Authorization header cf. autoproxy by Pat Thoyts.
# - Rewritten as a call to http::geturl, because response headers and body are
#   needed if the CONNECT request fails.  CONNECT is implemented for this case
#   only, by state(bypass).
# - FUTURE WORK: give http::geturl a -connect option for a general CONNECT.
# - The request header Proxy-Connection is discouraged in RFC 7230 (June 2014),
#   RFC 9112 (June 2022).
#
# Arguments:
# args        - as for ::socket, ending in host, port; with proxy host, proxy
#               port appended.
#
# Return Value: a socket identifier
# ------------------------------------------------------------------------------

proc http::SecureProxyConnect {args} {
    variable http
    variable ConnectVar
    variable ConnectCounter
    variable failedProxyValues
    set varName ::http::ConnectVar([incr ConnectCounter])

    # Extract (non-proxy) target from args.
    set host [lindex $args end-3]
    set port [lindex $args end-2]
    set args [lreplace $args end-3 end-2]

    # Proxy server URL for connection.
    # This determines where the socket is opened.
    set phost [lindex $args end-1]
    set pport [lindex $args end]
    if {[string first : $phost] != -1} {
	# IPv6 address, wrap it in [] so we can append :pport
	set phost "\[${phost}\]"
    }
    set url http://${phost}:${pport}
    # Elements of args other than host and port are not used when
    # AsyncTransaction opens a socket.  Those elements are -async and the
    # -type $tokenName for the https transaction.  Option -async is used by
    # AsyncTransaction anyway, and -type $tokenName should not be
    # propagated: the proxy request adds its own -type value.

    set targ [lsearch -exact $args -type]
    if {$targ != -1} {
	# Record in the token that this is a proxy call.
	set token [lindex $args $targ+1]
	upvar 0 ${token} state
	set tim $state(-timeout)
	set state(proxyUsed) SecureProxyFailed
	# This value is overwritten with "SecureProxy" below if the CONNECT is
	# successful.  If it is unsuccessful, the socket will be closed
	# below, and so in this unsuccessful case there are no other transactions
	# whose (proxyUsed) must be updated.
    } else {
	set tim 0
    }
    if {$tim == 0} {
	# Do not use infinite timeout for the proxy.
	set tim 30000
    }

    # Prepare and send a CONNECT request to the proxy, using
    # code similar to http::geturl.
    set requestHeaders [list Host $host]
    lappend requestHeaders Connection keep-alive
    if {$http(-proxyauth) != {}} {
	lappend requestHeaders Proxy-Authorization $http(-proxyauth)
    }

    set token2 [CreateToken $url -keepalive 0 -timeout $tim \
	    -headers $requestHeaders -command [list http::AllDone $varName]]
    variable $token2
    upvar 0 $token2 state2

    # Kludges:
    # Setting this variable overrides the HTTP request line and also allows
    # -headers to override the Connection: header set by -keepalive.
    # The arguments "-keepalive 0" ensure that when Finish is called for an
    # unsuccessful request, the socket is always closed.
    set state2(bypass) "CONNECT $host:$port HTTP/1.1"

    AsyncTransaction $token2

    if {[info coroutine] ne {}} {
	# All callers in the http package are coroutines launched by
	# the event loop.
	# The cwait command requires a coroutine because it yields
	# to the caller; $varName is traced and the coroutine resumes
	# when the variable is written.
	cwait $varName
    } else {
	return -code error {code must run in a coroutine}
	# For testing with a non-coroutine caller outside the http package.
	# vwait $varName
    }
    unset $varName

    if {    ($state2(state) ne "complete")
	 || ($state2(status) ne "ok")
	 || (![string is integer -strict $state2(responseCode)])
    } {
	set msg {the HTTP request to the proxy server did not return a valid\
		and complete response}
	if {[info exists state2(error)]} {
	    append msg ": " [lindex $state2(error) 0]
	}
	cleanup $token2
	return -code error $msg
    }

    set code $state2(responseCode)

    if {($code >= 200) && ($code < 300)} {
	# All OK.  The caller in package tls will now call "tls::import $sock".
	# The cleanup command does not close $sock.
	# Other tidying was done in http::Event.

	# If this is a persistent socket, any other transactions that are
	# already marked to use the socket will have their (proxyUsed) updated
	# when http::OpenSocket calls http::ConfigureNewSocket.
	set state(proxyUsed) SecureProxy
	set sock $state2(sock)
	cleanup $token2
	return $sock
    }

    if {$targ != -1} {
	# Non-OK HTTP status code; token is known because option -type
	# (cf. targ) was passed through tcltls, and so the useful
	# parts of the proxy's response can be copied to state(*).
	# Do not copy state2(sock).
	# Return the proxy response to the caller of geturl.
	foreach name $failedProxyValues {
	    if {[info exists state2($name)]} {
		set state($name) $state2($name)
	    }
	}
	set state(connection) close
	set msg "proxy connect failed: $code"
	# - This error message will be detected by http::OpenSocket and will
	#   cause it to present the proxy's HTTP response as that of the
	#   original $token transaction, identified only by state(proxyUsed)
	#   as the response of the proxy.
	# - The cases where this would mislead the caller of http::geturl are
	#   given a different value of msg (below) so that http::OpenSocket will
	#   treat them as errors, but will preserve the $token array for
	#   inspection by the caller.
	# - Status code 305 (Proxy Required) was deprecated for security reasons
	#   in RFC 2616 (June 1999) and in any case should never be served by a
	#   proxy.
	# - Other 3xx responses from the proxy are inappropriate, and should not
	#   occur.
	# - A 401 response from the proxy is inappropriate, and should not
	#   occur.  It would be confusing if returned to the caller.

	if {($code >= 300) && ($code < 400)} {
	    set msg "the proxy server responded to the HTTP request with an\
		    inappropriate $code redirect"
	    set loc [responseHeaderValue $token2 location]
	    if {$loc ne {}} {
		append msg "to " $loc
	    }
	} elseif {($code == 401)} {
	    set msg "the proxy server responded to the HTTP request with an\
		    inappropriate 401 request for target-host credentials"
	}
    } else {
	set msg "connection to proxy failed with status code $code"
    }

    # - ${token2}(sock) has already been closed because -keepalive 0.
    # - Error return does not pass the socket ID to the
    #   $token transaction, which retains its socket placeholder.
    cleanup $token2
    return -code error $msg
}

proc http::AllDone {varName args} {
    set $varName done
    return
}


# ------------------------------------------------------------------------------
#  Proc http::AltSocket
# ------------------------------------------------------------------------------
# This command is a drop-in replacement for ::socket.
# Arguments and return value as for ::socket.
#
# Notes.
# - http::AltSocket is specified in place of ::socket by the definition of
#   urlTypes in the namespace header of this file (http.tcl).
# - The command makes a simple call to ::socket unless the user has called
#   http::config to change the value of -threadlevel from the default value 0.
# - For -threadlevel 1 or 2, if the Thread package is available, the command
#   waits in the event loop while the socket is opened in another thread.  This
#   is a workaround for bug [824251] - it prevents http::geturl from blocking
#   the event loop if the DNS lookup or server connection is slow.
# - FIXME Use a thread pool if connections are very frequent.
# - FIXME The peer thread can transfer the socket only to the main interpreter
#   in the present thread.  Therefore this code works only if this script runs
#   in the main interpreter.  In a child interpreter, the parent must alias a
#   command to ::http::AltSocket in the child, run http::AltSocket in the
#   parent, and then transfer the socket to the child.
# - The http::AltSocket command is simple, and can easily be replaced with an
#   alternative command that uses a different technique to open a socket while
#   entering the event loop.
# - Unexpected behaviour by thread::send -async (Thread 2.8.6).
#   An error in thread::send -async causes return of just the error message
#   (not the expected 3 elements), and raises a bgerror in the main thread.
#   Hence wrap the command with catch as a precaution.
# - Bug in Thread 2.8.8 - on Windows, read/write operations fail on a socket
#   moved from another thread by thread::transfer.
# ------------------------------------------------------------------------------

proc http::AltSocket {args} {
    variable ThreadVar
    variable ThreadCounter
    variable http

    LoadThreadIfNeeded

    set targ [lsearch -exact $args -type]
    if {$targ != -1} {
	set token [lindex $args $targ+1]
	set args [lreplace $args $targ $targ+1]
	upvar 0 $token state
    }

    if {$http(usingThread) && [info exists state] && $state(protoSockThread)} {
    } else {
	# Use plain "::socket".  This is the default.
	return [eval ::socket $args]
    }

    set defcmd ::socket
    set sockargs $args
    set script "
	set code \[catch {
	    [list proc ::SockInThread {caller defcmd sockargs} [info body ::http::SockInThread]]
	    [list ::SockInThread [thread::id] $defcmd $sockargs]
	} result opts\]
	list \$code \$opts \$result
    "

    set state(tid) [thread::create]
    set varName ::http::ThreadVar([incr ThreadCounter])
    thread::send -async $state(tid) $script $varName
    Log >T Thread Start Wait $args -- coro [info coroutine] $varName
    if {[info coroutine] ne {}} {
	# All callers in the http package are coroutines launched by
	# the event loop.
	# The cwait command requires a coroutine because it yields
	# to the caller; $varName is traced and the coroutine resumes
	# when the variable is written.
	cwait $varName
    } else {
	return -code error {code must run in a coroutine}
	# For testing with a non-coroutine caller outside the http package.
	# vwait $varName
    }
    Log >U Thread End Wait $args -- coro [info coroutine] $varName [set $varName]
    thread::release $state(tid)
    set state(tid) {}
    set result [set $varName]
    unset $varName
    if {(![string is list $result]) || ([llength $result] != 3)} {
	return -code error "result from peer thread is not a list of\
		length 3: it is \n$result"
    }
    lassign $result threadCode threadDict threadResult
    if {($threadCode != 0)} {
	# This is an error in thread::send.  Return the lot.
	return -options $threadDict -code error $threadResult
    }

    # Now the results of the catch in the peer thread.
    lassign $threadResult catchCode errdict sock

    if {($catchCode == 0) && ($sock ni [chan names])} {
	return -code error {Transfer of socket from peer thread failed.\
		Check that this script is not running in a child interpreter.}
    }
    return -options $errdict -code $catchCode $sock
}

# The commands below are dependencies of http::AltSocket and
# http::SecureProxyConnect and are not used elsewhere.

# ------------------------------------------------------------------------------
#  Proc http::LoadThreadIfNeeded
# ------------------------------------------------------------------------------
# Command to load the Thread package if it is needed.  If it is needed and not
# loadable, the outcome depends on $http(-threadlevel):
# value 0 => Thread package not required, no problem
# value 1 => operate as if -threadlevel 0
# value 2 => error return
#
# The command assigns a value to http(usingThread), which records whether
# command http::AltSocket can use a separate thread.
#
# Arguments: none
# Return Value: none
# ------------------------------------------------------------------------------

proc http::LoadThreadIfNeeded {} {
    variable http
    if {$http(-threadlevel) == 0} {
	set http(usingThread) 0
	return
    }
    if {[catch {package require Thread}]} {
	if {$http(-threadlevel) == 2} {
	    set msg {[http::config -threadlevel] has value 2,\
		     but the Thread package is not available}
	    return -code error $msg
	}
	set http(usingThread) 0
	return
    }
    set http(usingThread) 1
    return
}


# ------------------------------------------------------------------------------
#  Proc http::SockInThread
# ------------------------------------------------------------------------------
# Command http::AltSocket is a ::socket replacement.  It defines and runs this
# command, http::SockInThread, in a peer thread.
#
# Arguments:
# caller
# defcmd
# sockargs
#
# Return value: list of values that describe the outcome.  The return is
# intended to be a normal (non-error) return in all cases.
# ------------------------------------------------------------------------------

proc http::SockInThread {caller defcmd sockargs} {
    package require Thread

    set catchCode [catch {eval $defcmd $sockargs} sock errdict]
    if {$catchCode == 0} {
	set catchCode [catch {thread::transfer $caller $sock; set sock} sock errdict]
    }
    return [list $catchCode $errdict $sock]
}


# ------------------------------------------------------------------------------
#  Proc http::cwaiter::cwait
# ------------------------------------------------------------------------------
# Command to substitute for vwait, without the ordering issues.
# A command that uses cwait must be a coroutine that is launched by an event,
# e.g. fileevent or after idle, and has no calling code to be resumed upon
# "yield".  It cannot return a value.
#
# Arguments:
# varName      - fully-qualified name of the variable that the calling script
#                will write to resume the coroutine.  Any scalar variable or
#                array element is permitted.
# coroName     - (optional) name of the coroutine to be called when varName is
#                written - defaults to this coroutine
# timeout      - (optional) timeout value in ms
# timeoutValue - (optional) value to assign to varName if there is a timeout
#
# Return Value: none
# ------------------------------------------------------------------------------

namespace eval http::cwaiter {
    namespace export cwait
    variable log   {}
    variable logOn 0
}

proc http::cwaiter::cwait {
    varName {coroName {}} {timeout {}} {timeoutValue {}}
} {
    set thisCoro [info coroutine]
    if {$thisCoro eq {}} {
	return -code error {cwait cannot be called outside a coroutine}
    }
    if {$coroName eq {}} {
	set coroName $thisCoro
    }
    if {[string range $varName 0 1] ne {::}} {
	return -code error {argument varName must be fully qualified}
    }
    if {$timeout eq {}} {
	set toe {}
    } elseif {[string is integer -strict $timeout] && ($timeout > 0)} {
	set toe [after $timeout [list set $varName $timeoutValue]]
    } else {
	return -code error {if timeout is supplied it must be a positive integer}
    }

    set cmd [list ::http::cwaiter::CwaitHelper $varName $coroName $toe]
    trace add variable $varName write $cmd
    CoLog "Yield $varName $coroName"
    yield
    CoLog "Resume $varName $coroName"
    return
}


# ------------------------------------------------------------------------------
#  Proc http::cwaiter::CwaitHelper
# ------------------------------------------------------------------------------
# Helper command called by the trace set by cwait.
# - Ignores the arguments added by trace.
# - A simple call to $coroName works, and in error cases gives a suitable stack
#   trace, but because it is inside a trace the headline error message is
#   something like {can't set "::Result(6)": error}, not the actual
#   error.  So let the trace command return.
# - Remove the trace immediately.  We don't want multiple calls.
# ------------------------------------------------------------------------------

proc http::cwaiter::CwaitHelper {varName coroName toe args} {
    CoLog "got $varName for $coroName"
    set cmd [list ::http::cwaiter::CwaitHelper $varName $coroName $toe]
    trace remove variable $varName write $cmd
    after cancel $toe

    after 0 $coroName
    return
}


# ------------------------------------------------------------------------------
#  Proc http::cwaiter::LogInit
# ------------------------------------------------------------------------------
# Call this command to initiate debug logging and clear the log.
# ------------------------------------------------------------------------------

proc http::cwaiter::LogInit {} {
    variable log
    variable logOn
    set log {}
    set logOn 1
    return
}

proc http::cwaiter::LogRead {} {
    variable log
    return $log
}

proc http::cwaiter::CoLog {msg} {
    variable log
    variable logOn
    if {$logOn} {
	append log $msg \n
    }
    return
}

namespace eval http {
    namespace import ::http::cwaiter::*
}

# Local variables:
# indent-tabs-mode: t
# End:
