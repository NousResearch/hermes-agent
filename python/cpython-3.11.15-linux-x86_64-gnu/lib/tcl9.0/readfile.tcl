# readFile:
# Read the contents of a file.
#
# Copyright © 2023 Donal K Fellows.
#
# See the file "license.terms" for information on usage and redistribution
# of this file, and for a DISCLAIMER OF ALL WARRANTIES.
#

proc readFile {filename {mode text}} {
    # Parse the arguments
    set MODES {binary text}
    set ERR [list -level 1 -errorcode [list TCL LOOKUP MODE $mode]]
    set mode [tcl::prefix match -message "mode" -error $ERR $MODES $mode]

    # Read the file
    set f [open $filename [dict get {text r binary rb} $mode]]
    try {
	return [read $f]
    } finally {
	close $f
    }
}
