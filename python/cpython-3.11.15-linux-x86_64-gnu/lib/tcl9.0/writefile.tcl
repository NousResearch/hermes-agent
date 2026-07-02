# writeFile:
# Write the contents of a file.
#
# Copyright © 2023 Donal K Fellows.
#
# See the file "license.terms" for information on usage and redistribution
# of this file, and for a DISCLAIMER OF ALL WARRANTIES.
#

proc writeFile {args} {
    # Parse the arguments
    switch [llength $args] {
	2 {
	    lassign $args filename data
	    set mode text
	}
	3 {
	    lassign $args filename mode data
	    set MODES {binary text}
	    set ERR [list -level 1 -errorcode [list TCL LOOKUP MODE $mode]]
	    set mode [tcl::prefix match -message "mode" -error $ERR $MODES $mode]
	}
	default {
	    set COMMAND [lindex [info level 0] 0]
	    return -code error -errorcode {TCL WRONGARGS} \
		"wrong # args: should be \"$COMMAND filename ?mode? data\""
	}
    }

    # Write the file
    set f [open $filename [dict get {text w binary wb} $mode]]
    try {
	puts -nonewline $f $data
    } finally {
	close $f
    }
}
