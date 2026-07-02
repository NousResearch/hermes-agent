#----------------------------------------------------------------------
#
# icu.tcl --
#
#	This file implements the portions of the [tcl::unsupported::icu]
#       ensemble that are coded in Tcl.
#
#----------------------------------------------------------------------
#
# Copyright © 2024 Ashok P. Nadkarni
# See the file "license.terms" for information on usage and redistribution
# of this file, and for a DISCLAIMER OF ALL WARRANTIES.
#
#----------------------------------------------------------------------

::tcl::unsupported::loadIcu

namespace eval ::tcl::unsupported::icu {
    # Map Tcl encoding names to ICU and back. Note ICU has multiple aliases
    # for the same encoding.
    variable tclToIcu
    variable icuToTcl

    proc LogError {message} {
	puts stderr $message
    }

    proc Init {} {
	variable tclToIcu
	variable icuToTcl
	# There are some special cases where names do not line up
	# at all. Map Tcl -> ICU
	array set specialCases {
	    ebcdic ebcdic-cp-us
	    macCentEuro maccentraleurope
	    utf16 UTF16_PlatformEndian
	    utf-16be UnicodeBig
	    utf-16le UnicodeLittle
	    utf32 UTF32_PlatformEndian
	}
	# Ignore all errors. Do not want to hold up Tcl
	# if ICU not available
	if {[catch {
	    foreach tclName [encoding names] {
		if {[catch {
		    set icuNames [aliases $tclName]
		} erMsg]} {
		    LogError "Could not get aliases for $tclName: $erMsg"
		    continue
		}
		if {[llength $icuNames] == 0} {
		    # E.g. macGreek -> x-MacGreek
		    set icuNames [aliases x-$tclName]
		    if {[llength $icuNames] == 0} {
			# Still no joy, check for special cases
			if {[info exists specialCases($tclName)]} {
			    set icuNames [aliases $specialCases($tclName)]
			}
		    }
		}
		# If the Tcl name is also an ICU name use it else use
		# the first name which is the canonical ICU name
		set pos [lsearch -exact -nocase $icuNames $tclName]
		if {$pos >= 0} {
		    lappend tclToIcu($tclName) [lindex $icuNames $pos] {*}[lreplace $icuNames $pos $pos]
		} else {
		    set tclToIcu($tclName) $icuNames
		}
		foreach icuName $icuNames {
		    lappend icuToTcl($icuName) $tclName
		}
	    }
	} errMsg]} {
	    LogError $errMsg
	}
	array default set tclToIcu ""
	array default set icuToTcl ""

	# Redefine ourselves to no-op.
	proc Init {} {}
    }
    # Primarily used during development
    proc MappedIcuNames {{pat *}} {
	Init
	variable icuToTcl
	return [array names icuToTcl $pat]
    }
    # Primarily used during development
    proc UnmappedIcuNames {{pat *}} {
	Init
	variable icuToTcl
	set unmappedNames {}
	foreach icuName [converters] {
	    if {[llength [icuToTcl $icuName]] == 0} {
		lappend unmappedNames $icuName
	    }
	    foreach alias [aliases $icuName] {
		if {[llength [icuToTcl $alias]] == 0} {
		    lappend unmappedNames $alias
		}
	    }
	}
	# Aliases can be duplicates. Remove
	return [lsort -unique [lsearch -inline -all $unmappedNames $pat]]
    }
    # Primarily used during development
    proc UnmappedTclNames {{pat *}} {
	Init
	variable tclToIcu
	set unmappedNames {}
	foreach tclName [encoding names] {
	    # Note entry will always exist. Check if empty
	    if {[llength [tclToIcu $tclName]] == 0} {
		lappend unmappedNames $tclName
	    }
	}
	return [lsearch -inline -all $unmappedNames $pat]
    }

    # Returns the Tcl equivalent of an ICU encoding name or
    # the empty string in case not found.
    proc icuToTcl {icuName} {
	Init
	proc icuToTcl {icuName} {
	    variable icuToTcl
	    return [lindex $icuToTcl($icuName) 0]
	}
	icuToTcl $icuName
    }

    # Returns the ICU equivalent of an Tcl encoding name or
    # the empty string in case not found.
    proc tclToIcu {tclName} {
	Init
	proc tclToIcu {tclName} {
	    variable tclToIcu
	    return [lindex $tclToIcu($tclName) 0]
	}
	tclToIcu $tclName
    }


    namespace export {[a-z]*}
    namespace ensemble create
}
