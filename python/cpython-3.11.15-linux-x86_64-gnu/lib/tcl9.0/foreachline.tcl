# foreachLine:
# Iterate over the contents of a file, a line at a time.
# The body script is run for each, with variable varName set to the line
# contents.
#
# Copyright © 2023 Donal K Fellows.
#
# See the file "license.terms" for information on usage and redistribution
# of this file, and for a DISCLAIMER OF ALL WARRANTIES.
#

proc foreachLine {varName filename body} {
    upvar 1 $varName line
    set f [open $filename "r"]
    try {
	while {[gets $f line] >= 0} {
	    uplevel 1 $body
	}
    } on return {msg opt} {
	dict incr opt -level
	return -options $opt $msg
    } finally {
	close $f
    }
}
