#
# itclWidget.tcl
# ----------------------------------------------------------------------
# Invoked automatically upon startup to customize the interpreter
# for [incr Tcl] when one of ::itcl::widget or ::itcl::widgetadaptor is called.
# ----------------------------------------------------------------------
#   AUTHOR:  Arnulf P. Wiedemann
#
# ----------------------------------------------------------------------
#            Copyright (c) 2008  Arnulf P. Wiedemann
# ======================================================================
# See the file "license.terms" for information on usage and
# redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.

package require Tk 8.6 9
# package require itclwidget [set ::itcl::version]

namespace eval ::itcl {

proc widget {name args} {
    set result [uplevel 1 ::itcl::internal::commands::genericclass widget $name $args]
    # we handle create by owerselfs !! allow classunknown to handle that
    oo::objdefine $result unexport create
    return $result
}

proc widgetadaptor {name args} {
    set result [uplevel 1 ::itcl::internal::commands::genericclass widgetadaptor $name $args]
    # we handle create by owerselfs !! allow classunknown to handle that
    oo::objdefine $result unexport create
    return $result
}

} ; # end ::itcl


namespace eval ::itcl::internal::commands {

proc initWidgetOptions {varNsName widgetName className} {
    set myDict [set ::itcl::internal::dicts::classOptions]
    if {$myDict eq ""} {
	return
    }
    if {![dict exists $myDict $className]} {
	return
    }
    set myDict [dict get $myDict $className]
    foreach option [dict keys $myDict] {
	set infos [dict get $myDict $option]
	set resource [dict get $infos -resource]
	set class [dict get $infos -class]
	set value [::option get $widgetName $resource $class]
	if {$value eq ""} {
	    if {[dict exists $infos -default]} {
		set defaultValue [dict get $infos -default]
		uplevel 1 set ${varNsName}::itcl_options($option) $defaultValue
	    }
	} else {
	    uplevel 1 set ${varNsName}::itcl_options($option) $value
	}
    }
}

proc initWidgetDelegatedOptions {varNsName widgetName className args} {
    set myDict [set ::itcl::internal::dicts::classDelegatedOptions]
    if {$myDict eq ""} {
	return
    }
    if {![dict exists $myDict $className]} {
	return
    }
    set myDict [dict get $myDict $className]
    foreach option [dict keys $myDict] {
	set infos [dict get $myDict $option]
	if {![dict exists $infos -resource]} {
	    # this is the case when delegating "*"
	    continue
	}
	if {![dict exists $infos -component]} {
	    # nothing to do
	    continue
	}
	# check if not in the command line options
	# these have higher priority
	set myOption $option
	if {[dict exists $infos -as]} {
	   set myOption [dict get $infos -as]
	}
	set noOptionSet 0
	foreach {optName optVal} $args {
	    if {$optName eq $myOption} {
		set noOptionSet 1
		break
	    }
	}
	if {$noOptionSet} {
	    continue
	}
	set resource [dict get $infos -resource]
	set class [dict get $infos -class]
	set component [dict get $infos -component]
	set value [::option get $widgetName $resource $class]
	if {$component ne ""} {
	    if {$value ne ""} {
		set compVar [namespace eval ${varNsName}${className} "set $component"]
		if {$compVar ne ""} {
		    uplevel 1 $compVar configure $myOption $value
		}
	    }
	}
    }
}

proc widgetinitobjectoptions {varNsName widgetName className} {
#puts stderr "initWidgetObjectOptions!$varNsName!$widgetName!$className!"
}

proc deletehull {newName oldName what} {
    if {$what eq "delete"} {
	set name [namespace tail $newName]
	regsub {hull[0-9]+} $name {} name
	rename $name {}
    }
    if {$what eq "rename"} {
	set name [namespace tail $newName]
	regsub {hull[0-9]+} $name {} name
	rename $name {}
    }
}

proc hullandoptionsinstall {objectName className widgetClass hulltype args} {
    if {$hulltype eq ""} {
	set hulltype frame
    }
    set idx 0
    set found 0
    foreach {optName optValue} $args {
	if {$optName eq "-class"} {
	    set found 1
	    set widgetClass $optValue
	    break
	}
	incr idx
    }
    if {$found} {
	set args [lreplace $args $idx [expr {$idx + 1}]]
    }
    if {$widgetClass eq ""} {
	set widgetClass $className
	set widgetClass [string totitle $widgetClass]
    }
    set cmd "set win $objectName; ::itcl::builtin::installhull using $hulltype -class $widgetClass $args"
    uplevel 2 $cmd
}

} ; # end ::itcl::internal::commands

namespace eval ::itcl::builtin {

proc installhull {args} {
    set cmdPath ::itcl::internal::commands
    set className [uplevel 1 info class]

    set replace 0
    switch -- [llength $args] {
	0	{
		return -code error\
		"wrong # args: should be \"[lindex [info level 0] 0]\
		name|using <widgetType> ?arg ...?\""
	}
	1	{
		set widgetName [lindex $args 0]
		set varNsName $::itcl::internal::varNsName($widgetName)
	}
	default	{
		upvar win win
		set widgetName $win

		set varNsName $::itcl::internal::varNsName($widgetName)
		set widgetType [lindex $args 1]
		incr replace
		if {[llength $args] > 3 && [lindex $args 2] eq "-class"} {
		    set classNam [lindex $args 3]
		    incr replace 2
		} else {
		    set classNam [string totitle $widgetType]
		}
		uplevel 1 [lreplace $args 0 $replace $widgetType $widgetName -class $classNam]
		uplevel 1 [list ${cmdPath}::initWidgetOptions $varNsName $widgetName $className]
	}
    }

    # initialize the itcl_hull variable
    set i 0
    set nam ::itcl::internal::widgets::hull
    while {1} {
	 incr i
	 set hullNam ${nam}${i}$widgetName
	 if {[::info command $hullNam] eq ""} {
	     break
	}
    }
    uplevel 1 [list ${cmdPath}::sethullwindowname $widgetName]
    uplevel 1 [list ::rename $widgetName $hullNam]
    uplevel 1 [list ::trace add command $hullNam {delete rename} ::itcl::internal::commands::deletehull]
    catch {${cmdPath}::checksetitclhull [list] 0}
    namespace eval ${varNsName}${className} "set itcl_hull $hullNam"
    catch {${cmdPath}::checksetitclhull [list] 2}
    uplevel 1 [lreplace $args 0 $replace ${cmdPath}::initWidgetDelegatedOptions $varNsName $widgetName $className]
}

proc installcomponent {args} {
    upvar win win

    set className [uplevel 1 info class]
    set myType [${className}::info types [namespace tail $className]]
    set isType 0
    if {$myType ne ""} {
	set isType 1
    }
    set numArgs [llength $args]
    set usage "usage: installcomponent <componentName> using <widgetType> <widgetPath> ?-option value ...?"
    if {$numArgs < 4} {
	error $usage
    }
    foreach {componentName using widgetType widgetPath} $args break
    set opts [lrange $args 4 end]
    if {$using ne "using"} {
	error $usage
    }
    if {!$isType} {
	set hullExists [uplevel 1 ::info exists itcl_hull]
	if {!$hullExists} {
	    error "cannot install \"$componentName\" before \"itcl_hull\" exists"
	}
	set hullVal [uplevel 1 set itcl_hull]
	if {$hullVal eq ""} {
	    error "cannot install \"$componentName\" before \"itcl_hull\" exists"
	}
    }
    # check for delegated option and ask the option database for the values
    # first check for number of delegated options
    set numOpts 0
    set starOption 0
    set myDict [set ::itcl::internal::dicts::classDelegatedOptions]
    if {[dict exists $myDict $className]} {
	set myDict [dict get $myDict $className]
	foreach option [dict keys $myDict] {
	    if {$option eq "*"} {
		set starOption 1
	    }
	    incr numOpts
	}
    }
    set myOptionDict [set ::itcl::internal::dicts::classOptions]
    if {[dict exists $myOptionDict $className]} {
	set myOptionDict [dict get $myOptionDict $className]
    }
    set cmd [list $widgetPath configure]
    set cmd1 "set $componentName \[$widgetType $widgetPath\]"
    uplevel 1 $cmd1
    if {$starOption} {
	upvar $componentName compName
	set cmd1 [list $compName configure]
	set configInfos [uplevel 1 $cmd1]
	foreach entry $configInfos {
	    if {[llength $entry] > 2} {
		foreach {optName resource class defaultValue} $entry break
		set val ""
		catch {
		    set val [::option get $win $resource $class]
		}
		if {$val ne ""} {
		    set addOpt 1
		    if {[dict exists $myDict $$optName]} {
			set addOpt 0
		    } else {
			set starDict [dict get $myDict "*"]
			if {[dict exists $starDict -except]} {
			    set exceptions [dict get $starDict -except]
			    if {[lsearch $exceptions $optName] >= 0} {
				set addOpt 0
			    }

			}
			if {[dict exists $myOptionDict $optName]} {
			    set addOpt 0
			}
		    }
		    if {$addOpt} {
			lappend cmd $optName $val
		    }

		}

	    }
	}
    } else {
	foreach optName [dict keys $myDict] {
	    set optInfos [dict get $myDict $optName]
	    set resource [dict get $optInfos -resource]
	    set class [namespace tail $className]
	    set class [string totitle $class]
	    set val ""
	    catch {
		set val [::option get $win $resource $class]
	    }
	    if {$val ne ""} {
		if {[dict exists $optInfos -as] } {
		    set optName [dict get $optInfos -as]
		}
		lappend cmd $optName $val
	    }
	}
    }
    lappend cmd {*}$opts
    uplevel 1 $cmd
}

} ; # end ::itcl::builtin

set ::itcl::internal::dicts::hullTypes [list \
       frame \
       toplevel \
       labelframe \
       ttk:frame \
       ttk:toplevel \
       ttk:labelframe \
    ]

namespace eval ::itcl::builtin::Info {

proc hulltypes {args} {
    namespace upvar ::itcl::internal::dicts hullTypes hullTypes

    set numArgs [llength $args]
    if {$numArgs > 1} {
	error "wrong # args should be: info hulltypes ?<pattern>?"
    }
    set pattern ""
    if {$numArgs > 0} {
	set pattern [lindex $args 0]
    }
    if {$pattern ne ""} {
	return [lsearch -all -inline -glob $hullTypes $pattern]
    }
    return $hullTypes

}

proc widgetclasses {args} {
    set numArgs [llength $args]
    if {$numArgs > 1} {
	error "wrong # args should be: info widgetclasses ?<pattern>?"
    }
    set pattern ""
    if {$numArgs > 0} {
	set pattern [lindex $args 0]
    }
    set myDict [set ::itcl::internal::dicts::classes]
    if {![dict exists $myDict widget]} {
	return [list]
    }
    set myDict [dict get $myDict widget]
    set result [list]
    if {$pattern ne ""} {
	foreach key [dict keys $myDict] {
	    set myInfo [dict get $myDict $key]
	    set value [dict get $myInfo -widget]
	    if {[string match $pattern $value]} {
		lappend result $value
	    }
	}
    } else {
	foreach key [dict keys $myDict] {
	    set myInfo [dict get $myDict $key]
	    lappend result [dict get $myInfo -widget]
	}
    }
    return $result
}

proc widgets {args} {
    set numArgs [llength $args]
    if {$numArgs > 1} {
	error "wrong # args should be: info widgets ?<pattern>?"
    }
    set pattern ""
    if {$numArgs > 0} {
	set pattern [lindex $args 0]
    }
    set myDict [set ::itcl::internal::dicts::classes]
    if {![dict exists $myDict widget]} {
	return [list]
    }
    set myDict [dict get $myDict widget]
    set result [list]
    if {$pattern ne ""} {
	foreach key [dict keys $myDict] {
	    set myInfo [dict get $myDict $key]
	    set value [dict get $myInfo -name]
	    if {[string match $pattern $value]} {
		lappend result $value
	    }
	}
    } else {
	foreach key [dict keys $myDict] {
	    set myInfo [dict get $myDict $key]
	    lappend result [dict get $myInfo -name]
	}
    }
    return $result
}

proc widgetadaptors {args} {
    set numArgs [llength $args]
    if {$numArgs > 1} {
	error "wrong # args should be: info widgetadaptors ?<pattern>?"
    }
    set pattern ""
    if {$numArgs > 0} {
	set pattern [lindex $args 0]
    }
    set myDict [set ::itcl::internal::dicts::classes]
    if {![dict exists $myDict widgetadaptor]} {
	return [list]
    }
    set myDict [dict get $myDict widgetadaptor]
    set result [list]
    if {$pattern ne ""} {
	foreach key [dict keys $myDict] {
	    set myInfo [dict get $myDict $key]
	    set value [dict get $myInfo -name]
	    if {[string match $pattern $value]} {
		lappend result $value
	    }
	}
    } else {
	foreach key [dict keys $myDict] {
	    set myInfo [dict get $myDict $key]
	    lappend result [dict get $myInfo -name]
	}
    }
    return $result
}

} ; # end ::itcl::builtin::Info
