#
# itcl.tcl
# ----------------------------------------------------------------------
# Invoked automatically upon startup to customize the interpreter
# for [incr Tcl].
# ----------------------------------------------------------------------
#   AUTHOR:  Michael J. McLennan
#            Bell Labs Innovations for Lucent Technologies
#            mmclennan@lucent.com
#            http://www.tcltk.com/itcl
# ----------------------------------------------------------------------
#            Copyright (c) 1993-1998  Lucent Technologies, Inc.
# ======================================================================
# See the file "license.terms" for information on usage and
# redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.

proc ::itcl::delete_helper { name args } {
    ::itcl::delete object $name
}

# ----------------------------------------------------------------------
#  USAGE:  local <className> <objName> ?<arg> <arg>...?
#
#  Creates a new object called <objName> in class <className>, passing
#  the remaining <arg>'s to the constructor.  Unlike the usual
#  [incr Tcl] objects, however, an object created by this procedure
#  will be automatically deleted when the local call frame is destroyed.
#  This command is useful for creating objects that should only remain
#  alive until a procedure exits.
# ----------------------------------------------------------------------
proc ::itcl::local {class name args} {
    set ptr [uplevel [list $class $name] $args]
    uplevel [list set itcl-local-$ptr $ptr]
    set cmd [uplevel namespace which -command $ptr]
    uplevel [list trace add variable itcl-local-$ptr unset \
	"::itcl::delete_helper $cmd"]
    return $ptr
}

# ----------------------------------------------------------------------
# auto_mkindex
# ----------------------------------------------------------------------
# Define Itcl commands that will be recognized by the auto_mkindex
# parser in Tcl...
#

#
# USAGE:  itcl::class name body
# Adds an entry for the given class declaration.
#
foreach __cmd {itcl::class class itcl::type type ictl::widget widget itcl::widgetadaptor widgetadaptor itcl::extendedclass extendedclass} {
    auto_mkindex_parser::command $__cmd {name body} {
	variable index
	variable scriptFile
	append index "set [list auto_index([fullname $name])]"
	append index " \[list source -encoding utf-8 \[file join \$dir [list $scriptFile]\]\]\n"

	variable parser
	variable contextStack
	set contextStack [linsert $contextStack 0 $name]
	$parser eval $body
	set contextStack [lrange $contextStack 1 end]
    }
}

#
# USAGE:  itcl::body name arglist body
# Adds an entry for the given method/proc body.
#
foreach __cmd {itcl::body body} {
    auto_mkindex_parser::command $__cmd {name arglist body} {
	variable index
	variable scriptFile
	append index "set [list auto_index([fullname $name])]"
	append index " \[list source -encoding utf-8 \[file join \$dir [list $scriptFile]\]\]\n"
    }
}

#
# USAGE:  itcl::configbody name arglist body
# Adds an entry for the given method/proc body.
#
foreach __cmd {itcl::configbody configbody} {
    auto_mkindex_parser::command $__cmd {name body} {
	variable index
	variable scriptFile
	append index "set [list auto_index([fullname $name])]"
	append index " \[list source -encoding utf-8 \[file join \$dir [list $scriptFile]\]\]\n"
    }
}

#
# USAGE:  ensemble name ?body?
# Adds an entry to the auto index list for the given ensemble name.
#
foreach __cmd {itcl::ensemble ensemble} {
    auto_mkindex_parser::command $__cmd {name {body ""}} {
	variable index
	variable scriptFile
	append index "set [list auto_index([fullname $name])]"
	append index " \[list source -encoding utf-8 \[file join \$dir [list $scriptFile]\]\]\n"
    }
}

#
# USAGE:  public arg ?arg arg...?
#         protected arg ?arg arg...?
#         private arg ?arg arg...?
#
# Evaluates the arguments as commands, so we can recognize proc
# declarations within classes.
#
foreach __cmd {public protected private} {
    auto_mkindex_parser::command $__cmd {args} {
	variable parser
	$parser eval $args
    }
}

# SF bug #246 unset variable __cmd to avoid problems in user programs!!
unset __cmd

# ----------------------------------------------------------------------
# auto_import
# ----------------------------------------------------------------------
# This procedure overrides the usual "auto_import" function in the
# Tcl library.  It is invoked during "namespace import" to make see
# if the imported commands reside in an autoloaded library.  If so,
# stubs are created to represent the commands.  Executing a stub
# later on causes the real implementation to be autoloaded.
#
# Arguments -
# pattern	The pattern of commands being imported (like "foo::*")
#               a canonical namespace as returned by [namespace current]

proc auto_import {pattern} {
    global auto_index

    set ns [uplevel namespace current]
    set patternList [auto_qualify $pattern $ns]

    auto_load_index

    foreach pattern $patternList {
	foreach name [array names auto_index $pattern] {
	    if {"" == [info commands $name]} {
		::itcl::import::stub create $name
	    }
	}
    }
}
