#
# ttrace.tcl --
#
# Copyright (C) 2003 Zoran Vasiljevic, Archiware GmbH. All Rights Reserved.
#
# See the file "license.terms" for information on usage and redistribution of
# this file, and for a DISCLAIMER OF ALL WARRANTIES.
# ----------------------------------------------------------------------------
#
# User level commands:
#
#   ttrace::eval           top-level wrapper (ttrace-savvy eval)
#   ttrace::enable         activates registered Tcl command traces
#   ttrace::disable        terminates tracing of Tcl commands
#   ttrace::isenabled      returns true if ttrace is enabled
#   ttrace::cleanup        bring the interp to a pristine state
#   ttrace::update         update interp to the latest trace epoch
#   ttrace::config         setup some configuration options
#   ttrace::getscript      returns a script for initializing interps
#
# Commands used for/from trace callbacks:
#
#   ttrace::atenable       register callback to be done at trace enable
#   ttrace::atdisable      register callback to be done at trace disable
#   ttrace::addtrace       register user-defined tracer callback
#   ttrace::addscript      register user-defined script generator
#   ttrace::addresolver    register user-defined command resolver
#   ttrace::addcleanup     register user-defined cleanup procedures
#   ttrace::addentry       adds one entry into the named trace store
#   ttrace::getentry       returns the entry value from the named store
#   ttrace::delentry       removes the entry from the named store
#   ttrace::getentries     returns all entries from the named store
#   ttrace::preload        register procedures to be preloaded always
#
#
# Limitations:
#
#   o. [namespace forget] is still not implemented
#   o. [namespace origin cmd] breaks if cmd is not already defined
#
#      I left this deliberately. I didn't want to override the [namespace]
#      command in order to avoid potential slowdown.
#

namespace eval ttrace {

    # Setup some compatibility wrappers
    if {[info commands nsv_set] != ""} {
	variable tvers 0
	variable mutex ns_mutex
	variable elock [$mutex create traceepochmutex]
	# Import the underlying API; faster than recomputing
	interp alias {} [namespace current]::_array   {} nsv_array
	interp alias {} [namespace current]::_incr    {} nsv_incr
	interp alias {} [namespace current]::_lappend {} nsv_lappend
	interp alias {} [namespace current]::_names   {} nsv_names
	interp alias {} [namespace current]::_set     {} nsv_set
	interp alias {} [namespace current]::_unset   {} nsv_unset
    } elseif {![catch {
	variable tvers [package require thread]
    }]} {
	variable mutex thread::mutex
	variable elock [$mutex create]
	# Import the underlying API; faster than recomputing
	interp alias {} [namespace current]::_array   {} tsv::array
	interp alias {} [namespace current]::_incr    {} tsv::incr
	interp alias {} [namespace current]::_lappend {} tsv::lappend
	interp alias {} [namespace current]::_names   {} tsv::names
	interp alias {} [namespace current]::_set     {} tsv::set
	interp alias {} [namespace current]::_unset   {} tsv::unset
    } else {
	error "requires NaviServer/AOLserver or Tcl threading extension"
    }

    # Keep in sync with the thread package
    package provide ttrace 3.0.4
    package provide Ttrace 3.0.4

    # Package variables
    variable resolvers ""     ; # List of registered resolvers
    variable tracers   ""     ; # List of registered cmd tracers
    variable scripts   ""     ; # List of registered script makers
    variable enables   ""     ; # List of trace-enable callbacks
    variable disables  ""     ; # List of trace-disable callbacks
    variable preloads  ""     ; # List of procedure names to preload
    variable enabled   0      ; # True if trace is enabled
    variable config           ; # Array with config options

    variable epoch     -1     ; # The initialization epoch
    variable cleancnt   0     ; # Counter of registered cleaners

    # Package private namespaces
    namespace eval resolve "" ; # Commands for resolving commands
    namespace eval trace   "" ; # Commands registered for tracing
    namespace eval enable  "" ; # Commands invoked at trace enable
    namespace eval disable "" ; # Commands invoked at trace disable
    namespace eval script  "" ; # Commands for generating scripts

    # Exported commands
    namespace export unknown

    # Initialize ttrace shared state
    if {[_array exists ttrace] == 0} {
	_set ttrace lastepoch $epoch
	_set ttrace epochlist ""
    }

    # Initially, allow creation of epochs
    set config(-doepochs) 1

    proc eval {cmd args} {
	enable
	set code [catch {uplevel 1 [concat $cmd $args]} result]
	disable
	if {$code == 0} {
	    if {[llength [info commands ns_ictl]]} {
		ns_ictl save [getscript]
	    } else {
		thread::broadcast {
		    package require ttrace
		    ttrace::update
		}
	    }
	}
	return -code $code \
	    -errorinfo $::errorInfo -errorcode $::errorCode $result
    }

    proc config {args} {
	variable config
	if {[llength $args] == 0} {
	    array get config
	} elseif {[llength $args] == 1} {
	    set opt [lindex $args 0]
	    set config($opt)
	} else {
	    set opt [lindex $args 0]
	    set val [lindex $args 1]
	    set config($opt) $val
	}
    }

    proc enable {} {
	variable config
	variable tracers
	variable enables
	variable enabled
	incr enabled 1
	if {$enabled > 1} {
	    return
	}
	if {$config(-doepochs) != 0} {
	    variable epoch [_newepoch]
	}
	set nsp [namespace current]
	foreach enabler $enables {
	    enable::_$enabler
	}
	foreach trace $tracers {
	    if {[info commands $trace] != ""} {
		trace add execution $trace leave ${nsp}::trace::_$trace
	    }
	}
    }

    proc disable {} {
	variable enabled
	variable tracers
	variable disables
	incr enabled -1
	if {$enabled > 0} {
	    return
	}
	set nsp [namespace current]
	foreach disabler $disables {
	    disable::_$disabler
	}
	foreach trace $tracers {
	    if {[info commands $trace] != ""} {
		trace remove execution $trace leave ${nsp}::trace::_$trace
	    }
	}
    }

    proc isenabled {} {
	variable enabled
	expr {$enabled > 0}
    }

    proc update {{from -1}} {
	if {$from < 0} {
	    variable epoch [_set ttrace lastepoch]
	} else {
	    if {[lsearch [_set ttrace epochlist] $from] < 0} {
		error "no such epoch: $from"
	    }
	    variable epoch $from
	}
	uplevel 1 [getscript]
    }

    proc getscript {} {
	variable preloads
	variable epoch
	variable scripts
	append script [_serializensp] \n
	append script "::namespace eval [namespace current] {" \n
	append script "::namespace export unknown" \n
	append script "_useepoch $epoch" \n
	append script "}" \n
	foreach cmd $preloads {
	    append script [_serializeproc $cmd] \n
	}
	foreach maker $scripts {
	    append script [script::_$maker]
	}
	return $script
    }

    proc cleanup {args} {
	foreach cmd [info commands resolve::cleaner_*] {
	    uplevel 1 $cmd $args
	}
    }

    proc preload {cmd} {
	variable preloads
	if {[lsearch $preloads $cmd] < 0} {
	    lappend preloads $cmd
	}
    }

    proc atenable {cmd arglist body} {
	variable enables
	if {[lsearch $enables $cmd] < 0} {
	    lappend enables $cmd
	    set cmd [namespace current]::enable::_$cmd
	    proc $cmd $arglist $body
	    return $cmd
	}
    }

    proc atdisable {cmd arglist body} {
	variable disables
	if {[lsearch $disables $cmd] < 0} {
	    lappend disables $cmd
	    set cmd [namespace current]::disable::_$cmd
	    proc $cmd $arglist $body
	    return $cmd
	}
    }

    proc addtrace {cmd arglist body} {
	variable tracers
	if {[lsearch $tracers $cmd] < 0} {
	    lappend tracers $cmd
	    set tracer [namespace current]::trace::_$cmd
	    proc $tracer $arglist $body
	    if {[isenabled]} {
		trace add execution $cmd leave $tracer
	    }
	    return $tracer
	}
    }

    proc addscript {cmd body} {
	variable scripts
	if {[lsearch $scripts $cmd] < 0} {
	    lappend scripts $cmd
	    set cmd [namespace current]::script::_$cmd
	    proc $cmd args $body
	    return $cmd
	}
    }

    proc addresolver {cmd arglist body} {
	variable resolvers
	if {[lsearch $resolvers $cmd] < 0} {
	    lappend resolvers $cmd
	    set cmd [namespace current]::resolve::$cmd
	    proc $cmd $arglist $body
	    return $cmd
	}
    }

    proc addcleanup {body} {
	variable cleancnt
	set cmd [namespace current]::resolve::cleaner_[incr cleancnt]
	proc $cmd args $body
	return $cmd
    }

    proc addentry {cmd var val} {
	variable epoch
	_set ${epoch}-$cmd $var $val
    }

    proc delentry {cmd var} {
	variable epoch
	set ei $::errorInfo
	set ec $::errorCode
	catch {_unset ${epoch}-$cmd $var}
	set ::errorInfo $ei
	set ::errorCode $ec
    }

    proc getentry {cmd var} {
	variable epoch
	set ei $::errorInfo
	set ec $::errorCode
	if {[catch {_set ${epoch}-$cmd $var} val]} {
	    set ::errorInfo $ei
	    set ::errorCode $ec
	    set val ""
	}
	return $val
    }

    proc getentries {cmd {pattern *}} {
	variable epoch
	_array names ${epoch}-$cmd $pattern
    }

    proc unknown {args} {
	set cmd [lindex $args 0]
	if {[uplevel 1 ttrace::_resolve [list $cmd]]} {
	    set c [catch {uplevel 1 $cmd [lrange $args 1 end]} r]
	} else {
	    set c [catch {uplevel 1 ::tcl::unknown $args} r]
	}
	return -code $c -errorcode $::errorCode -errorinfo $::errorInfo $r
    }

    proc _resolve {cmd} {
	variable resolvers
	foreach resolver $resolvers {
	    if {[uplevel 1 [info comm resolve::$resolver] [list $cmd]]} {
		return 1
	    }
	}
	return 0
    }

    proc _getthread {} {
	if {[info commands ns_thread] == ""} {
	    thread::id
	} else {
	    ns_thread getid
	}
    }

    proc _getthreads {} {
	if {[info commands ns_thread] == ""} {
	    return [thread::names]
	} else {
	    foreach entry [ns_info threads] {
		lappend threads [lindex $entry 2]
	    }
	    return $threads
	}
    }

    proc _newepoch {} {
	variable elock
	variable mutex
	$mutex lock $elock
	set old [_set ttrace lastepoch]
	set new [_incr ttrace lastepoch]
	_lappend ttrace $new [_getthread]
	if {$old >= 0} {
	    _copyepoch $old $new
	    _delepochs
	}
	_lappend ttrace epochlist $new
	$mutex unlock $elock
	return $new
    }

    proc _copyepoch {old new} {
	foreach var [_names $old-*] {
	    set cmd [lindex [split $var -] 1]
	    _array reset $new-$cmd [_array get $var]
	}
    }

    proc _delepochs {} {
	set tlist [_getthreads]
	set elist ""
	foreach epoch [_set ttrace epochlist] {
	    if {[_dropepoch $epoch $tlist] == 0} {
		lappend elist $epoch
	    } else {
		_unset ttrace $epoch
	    }
	}
	_set ttrace epochlist $elist
    }

    proc _dropepoch {epoch threads} {
	set self [_getthread]
	foreach tid [_set ttrace $epoch] {
	    if {$tid != $self && [lsearch $threads $tid] >= 0} {
		lappend alive $tid
	    }
	}
	if {[info exists alive]} {
	    _set ttrace $epoch $alive
	    return 0
	} else {
	    foreach var [_names $epoch-*] {
		_unset $var
	    }
	    return 1
	}
    }

    proc _useepoch {epoch} {
	if {$epoch >= 0} {
	    set tid [_getthread]
	    if {[lsearch [_set ttrace $epoch] $tid] == -1} {
		_lappend ttrace $epoch $tid
	    }
	}
    }

    proc _serializeproc {cmd} {
	set dargs [info args $cmd]
	set pbody [info body $cmd]
	set pargs ""
	foreach arg $dargs {
	    if {![info default $cmd $arg def]} {
		lappend pargs $arg
	    } else {
		lappend pargs [list $arg $def]
	    }
	}
	set nsp [namespace qual $cmd]
	if {$nsp == ""} {
	    set nsp "::"
	}
	append res [list ::namespace eval $nsp] " {" \n
	append res [list ::proc [namespace tail $cmd] $pargs $pbody] \n
	append res "}" \n
    }

    proc _serializensp {{nsp ""} {result _}} {
	upvar $result res
	if {$nsp == ""} {
	    set nsp [namespace current]
	}
	append res [list ::namespace eval $nsp] " {" \n
	foreach var [info vars ${nsp}::*] {
	    set vname [namespace tail $var]
	    if {[array exists $var] == 0} {
		append res [list ::variable $vname [set $var]] \n
	    } else {
		append res [list ::variable $vname] \n
		append res [list ::array set $vname [array get $var]] \n
	    }
	}
	foreach cmd [info procs ${nsp}::*] {
	    append res [_serializeproc $cmd] \n
	}
	append res "}" \n
	foreach nn [namespace children $nsp] {
	    _serializensp $nn res
	}
	return $res
    }
}

#
# The code below is ment to be run once during the application start.  It
# provides implementation of tracing callbacks for some Tcl commands.  Users
# can supply their own tracer implementations on-the-fly.
#
# The code below will create traces for the following Tcl commands:
#    "namespace", "variable", "load", "proc" and "rename"
#
# Also, the Tcl object extension XOTcl 1.1.0 is handled and all XOTcl related
# things, like classes and objects are traced (many thanks to Gustaf Neumann
# from XOTcl for his kind help and support).
#

eval {

    #
    # Register the "load" trace. This will create the following key/value pair
    # in the "load" store:
    #
    #  --- key ----              --- value ---
    #  <path_of_loaded_image>    <name_of_the_init_proc>
    #
    # We normally need only the name_of_the_init_proc for being able to load
    # the package in other interpreters, but we store the path to the image
    # file as well.
    #

    ttrace::addtrace load {cmdline code args} {
	if {$code != 0} {
	    return
	}
	set image [lindex $cmdline 1]
	set initp [lindex $cmdline 2]
	if {$initp == ""} {
	    foreach pkg [info loaded] {
		if {[lindex $pkg 0] == $image} {
		    set initp [lindex $pkg 1]
		}
	    }
	}
	ttrace::addentry load $image $initp
    }

    ttrace::addscript load {
	append res "\n"
	foreach entry [ttrace::getentries load] {
	    set initp [ttrace::getentry load $entry]
	    append res "::load {} $initp" \n
	}
	return $res
    }

    #
    # Register the "namespace" trace. This will create the following key/value
    # entry in "namespace" store:
    #
    #  --- key ----                   --- value ---
    #  ::fully::qualified::namespace  1
    #
    # It will also fill the "proc" store for procedures and commands imported
    # in this namespace with following:
    #
    #  --- key ----                   --- value ---
    #  ::fully::qualified::proc       [list <ns>  "" ""]
    #
    # The <ns> is the name of the namespace where the command or procedure is
    # imported from.
    #

    ttrace::addtrace namespace {cmdline code args} {
	if {$code != 0} {
	    return
	}
	set nop [lindex $cmdline 1]
	set cns [uplevel 1 namespace current]
	if {$cns == "::"} {
	    set cns ""
	}
	switch -glob $nop {
	    eva* {
		set nsp [lindex $cmdline 2]
		if {![string match "::*" $nsp]} {
		    set nsp ${cns}::$nsp
		}
		ttrace::addentry namespace $nsp 1
	    }
	    imp* {
		# - parse import arguments (skip opt "-force")
		set opts [lrange $cmdline 2 end]
		if {[string match "-fo*" [lindex $opts 0]]} {
		    set opts [lrange $cmdline 3 end]
		}
		# - register all imported procs and commands
		foreach opt $opts {
		    if {![string match "::*" [::namespace qual $opt]]} {
			set opt ${cns}::$opt
		    }
		    # - first import procs
		    foreach entry [ttrace::getentries proc $opt] {
			set cmd ${cns}::[::namespace tail $entry]
			set nsp [::namespace qual $entry]
			set done($cmd) 1
			set entry [list 0 $nsp "" ""]
			ttrace::addentry proc $cmd $entry
		    }

		    # - then import commands
		    foreach entry [info commands $opt] {
			set cmd ${cns}::[::namespace tail $entry]
			set nsp [::namespace qual $entry]
			if {[info exists done($cmd)] == 0} {
			    set entry [list 0 $nsp "" ""]
			    ttrace::addentry proc $cmd $entry
			}
		    }
		}
	    }
	}
    }

    ttrace::addscript namespace {
	append res \n
	foreach entry [ttrace::getentries namespace] {
	    append res "::namespace eval $entry {}" \n
	}
	return $res
    }

    #
    # Register the "variable" trace. This will create the following key/value
    # entry in the "variable" store:
    #
    #  --- key ----                   --- value ---
    #  ::fully::qualified::variable   1
    #
    # The variable value itself is ignored at the time of
    # trace/collection. Instead, we take the real value at the time of script
    # generation.
    #

    ttrace::addtrace variable {cmdline code args} {
	if {$code != 0} {
	    return
	}
	set opts [lrange $cmdline 1 end]
	if {[llength $opts]} {
	    set cns [uplevel 1 namespace current]
	    if {$cns == "::"} {
		set cns ""
	    }
	    foreach {var val} $opts {
		if {![string match "::*" $var]} {
		    set var ${cns}::$var
		}
		ttrace::addentry variable $var 1
	    }
	}
    }

    ttrace::addscript variable {
	append res \n
	foreach entry [ttrace::getentries variable] {
	    set cns [namespace qual $entry]
	    set var [namespace tail $entry]
	    append res "::namespace eval $cns {" \n
	    append res "::variable $var"
	    if {[array exists $entry]} {
		append res "\n::array set $var [list [array get $entry]]" \n
	    } elseif {[info exists $entry]} {
		append res " [list [set $entry]]" \n
	    } else {
		append res \n
	    }
	    append res "}" \n
	}
	return $res
    }


    #
    # Register the "rename" trace. It will create the following key/value pair
    # in "rename" store:
    #
    #  --- key ----              --- value ---
    #  ::fully::qualified::old  ::fully::qualified::new
    #
    # The "new" value may be empty, for commands that have been deleted. In
    # such cases we also remove any traced procedure definitions.
    #

    ttrace::addtrace rename {cmdline code args} {
	if {$code != 0} {
	    return
	}
	set cns [uplevel 1 namespace current]
	if {$cns == "::"} {
	    set cns ""
	}
	set old [lindex $cmdline 1]
	if {![string match "::*" $old]} {
	    set old ${cns}::$old
	}
	set new [lindex $cmdline 2]
	if {$new != ""} {
	    if {![string match "::*" $new]} {
		set new ${cns}::$new
	    }
	    ttrace::addentry rename $old $new
	} else {
	    ttrace::delentry proc $old
	}
    }

    ttrace::addscript rename {
	append res \n
	foreach old [ttrace::getentries rename] {
	    set new [ttrace::getentry rename $old]
	    append res "::rename $old {$new}" \n
	}
	return $res
    }

    #
    # Register the "proc" trace. This will create the following key/value pair
    # in the "proc" store:
    #
    #  --- key ----              --- value ---
    #  ::fully::qualified::proc  [list <epoch> <ns> <arglist> <body>]
    #
    # The <epoch> chages anytime one (re)defines a proc.  The <ns> is the
    # namespace where the command was imported from. If empty, the <arglist>
    # and <body> will hold the actual procedure definition. See the
    # "namespace" tracer implementation also.
    #

    ttrace::addtrace proc {cmdline code args} {
	if {$code != 0} {
	    return
	}
	set cns [uplevel 1 namespace current]
	if {$cns == "::"} {
	    set cns ""
	}
	set cmd [lindex $cmdline 1]
	if {![string match "::*" $cmd]} {
	    set cmd ${cns}::$cmd
	}
	set dargs [info args $cmd]
	set pbody [info body $cmd]
	set pargs ""
	foreach arg $dargs {
	    if {![info default $cmd $arg def]} {
		lappend pargs $arg
	    } else {
		lappend pargs [list $arg $def]
	    }
	}
	set pdef [ttrace::getentry proc $cmd]
	if {$pdef == ""} {
	    set epoch -1 ; # never traced before
	} else {
	    set epoch [lindex $pdef 0]
	}
	ttrace::addentry proc $cmd [list [incr epoch] "" $pargs $pbody]
    }

    ttrace::addscript proc {
	return {
	    if {[info command ::tcl::unknown] == ""} {
		rename ::unknown ::tcl::unknown
		namespace import -force ::ttrace::unknown
	    }
	    if {[info command ::tcl::info] == ""} {
		rename ::info ::tcl::info
	    }
	    proc ::info args {
		set cmd [lindex $args 0]
		set hit [lsearch -glob {commands procs args default body} $cmd*]
		if {$hit > 1} {
		    if {[catch {uplevel 1 ::tcl::info $args}]} {
			uplevel 1 ttrace::_resolve [list [lindex $args 1]]
		    }
		    return [uplevel 1 ::tcl::info $args]
		}
		if {$hit == -1} {
		    return [uplevel 1 ::tcl::info $args]
		}
		set cns [uplevel 1 namespace current]
		if {$cns == "::"} {
		    set cns ""
		}
		set pat [lindex $args 1]
		if {![string match "::*" $pat]} {
		    set pat ${cns}::$pat
		}
		set fns [ttrace::getentries proc $pat]
		if {[string match $cmd* commands]} {
		    set fns [concat $fns [ttrace::getentries xotcl $pat]]
		}
		foreach entry $fns {
		    if {$cns != [namespace qual $entry]} {
			set lazy($entry) 1
		    } else {
			set lazy([namespace tail $entry]) 1
		    }
		}
		foreach entry [uplevel 1 ::tcl::info $args] {
		    set lazy($entry) 1
		}
		array names lazy
	    }
	}
    }

    #
    # Register procedure resolver. This will try to resolve the command in the
    # current namespace first, and if not found, in global namespace.  It also
    # handles commands imported from other namespaces.
    #

    ttrace::addresolver resolveprocs {cmd {export 0}} {
	set cns [uplevel 1 namespace current]
	set name [namespace tail $cmd]
	if {$cns == "::"} {
	    set cns ""
	}
	if {![string match "::*" $cmd]} {
	    set ncmd ${cns}::$cmd
	    set gcmd ::$cmd
	} else {
	    set ncmd $cmd
	    set gcmd $cmd
	}
	set pdef [ttrace::getentry proc $ncmd]
	if {$pdef == ""} {
	    set pdef [ttrace::getentry proc $gcmd]
	    if {$pdef == ""} {
		return 0
	    }
	    set cmd $gcmd
	} else {
	    set cmd $ncmd
	}
	set epoch [lindex $pdef 0]
	set pnsp  [lindex $pdef 1]
	if {$pnsp != ""} {
	    set nsp [namespace qual $cmd]
	    if {$nsp == ""} {
		set nsp ::
	    }
	    set cmd ${pnsp}::$name
	    if {[resolveprocs $cmd 1] == 0 && [info commands $cmd] == ""} {
		return 0
	    }
	    namespace eval $nsp "namespace import -force $cmd"
	} else {
	    uplevel 0 [list ::proc $cmd [lindex $pdef 2] [lindex $pdef 3]]
	    if {$export} {
		set nsp [namespace qual $cmd]
		if {$nsp == ""} {
		    set nsp ::
		}
		namespace eval $nsp "namespace export $name"
	    }
	}
	variable resolveproc
	set resolveproc($cmd) $epoch
	return 1
    }

    #
    # For XOTcl, the entire item introspection/tracing is delegated to XOTcl
    # itself. The xotcl store is filled with this:
    #
    #  --- key ----               --- value ---
    #  ::fully::qualified::item   <body>
    #
    # The <body> is the script used to generate the entire item (class,
    # object). Note that we do not fill in this during code tracing. It is
    # done during the script generation. In this step, only the placeholder is
    # set.
    #
    # NOTE: we assume all XOTcl commands are imported in global namespace
    #

    ttrace::atenable XOTclEnabler {args} {
	if {[info commands ::xotcl::Class] == ""} {
	    return
	}
	if {[info commands ::xotcl::_creator] == ""} {
	    ::xotcl::Class create ::xotcl::_creator -instproc create {args} {
		set result [next]
		if {![string match ::xotcl::_* $result]} {
		    ttrace::addentry xotcl $result ""
		}
		return $result
	    }
	}
	::xotcl::Class instmixin ::xotcl::_creator
    }

    ttrace::atdisable XOTclDisabler {args} {
	if {   [info commands ::xotcl::Class] == ""
	    || [info commands ::xotcl::_creator] == ""} {
	    return
	}
	::xotcl::Class instmixin ""
	::xotcl::_creator destroy
    }

    set resolver [ttrace::addresolver resolveclasses {classname} {
	set cns [uplevel 1 namespace current]
	set script [ttrace::getentry xotcl $classname]
	if {$script == ""} {
	    set name [namespace tail $classname]
	    if {$cns == "::"} {
		set script [ttrace::getentry xotcl ::$name]
	    } else {
		set script [ttrace::getentry xotcl ${cns}::$name]
		if {$script == ""} {
		    set script [ttrace::getentry xotcl ::$name]
		}
	    }
	    if {$script == ""} {
		return 0
	    }
	}
	uplevel 1 [list namespace eval $cns $script]
	return 1
    }]

    ttrace::addscript xotcl [subst -nocommands {
	if {![catch {Serializer new} ss]} {
	    foreach entry [ttrace::getentries xotcl] {
		if {[ttrace::getentry xotcl \$entry] == ""} {
		    ttrace::addentry xotcl \$entry [\$ss serialize \$entry]
		}
	    }
	    \$ss destroy
	    return {::xotcl::Class proc __unknown name {$resolver \$name}}
	}
    }]

    #
    # Register callback to be called on cleanup. This will trash lazily loaded
    # procs which have changed since.
    #

    ttrace::addcleanup {
	variable resolveproc
	foreach cmd [array names resolveproc] {
	    set def [ttrace::getentry proc $cmd]
	    if {$def != ""} {
		set new [lindex $def 0]
		set old $resolveproc($cmd)
		if {[info command $cmd] != "" && $new != $old} {
		    catch {rename $cmd ""}
		}
	    }
	}
    }
}

# EOF
return

# Local Variables:
# mode: tcl
# fill-column: 78
# tab-width: 8
# indent-tabs-mode: nil
# End:
