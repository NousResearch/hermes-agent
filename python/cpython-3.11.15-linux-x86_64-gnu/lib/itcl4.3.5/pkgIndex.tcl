# -*- tcl -*-
# Tcl package index file, version 1.1
#

if {![package vsatisfies [package provide Tcl] 8.6-]} {return}

if {[package vsatisfies [package provide Tcl] 9.0-]} {
    package ifneeded itcl 4.3.5 \
	    [list load [file join $dir libtcl9itcl4.3.5.so] Itcl]
} else {
    package ifneeded itcl 4.3.5 \
	    [list load [file join $dir libitcl4.3.5.so] Itcl]
}
package ifneeded Itcl 4.3.5 [list package require -exact itcl 4.3.5]
