# -*- tcl -*-
# Tcl package index file, version 1.1
#

if {![package vsatisfies [package provide Tcl] 8.6-]} {
    return
}

#
# For interps that are not thread-enabled, we still call [package ifneeded].
# This is contrary to the usual convention, but is a good idea because we
# cannot imagine any other version of thread that might succeed in a
# thread-disabled interp.  There's nothing to gain by yielding to other
# competing callers of [package ifneeded Thread].  On the other hand,
# deferring the error has the advantage that a script calling
# [package require Thread] in a thread-disabled interp gets an error message
# about a thread-disabled interp, instead of the message
# "can't find package thread".

if {[package vsatisfies [package provide Tcl] 9.0-]} {
    package ifneeded thread 3.0.4 \
	    [list load [file join $dir libtcl9thread3.0.4.so] [string totitle thread]]
} else {
    package ifneeded thread 3.0.4 \
	    [list load [file join $dir libthread3.0.4.so] [string totitle thread]]
}
package ifneeded [string totitle thread] 3.0.4 \
    [list package require -exact thread 3.0.4]

# package ttrace uses some support machinery.

package ifneeded ttrace 3.0.4 [list ::apply {{dir} {
    if {[info exists ::env(TCL_THREAD_LIBRARY)]
	    && [file readable $::env(TCL_THREAD_LIBRARY)/ttrace.tcl]} {
	source $::env(TCL_THREAD_LIBRARY)/ttrace.tcl
    } elseif {[file readable [file join $dir .. lib ttrace.tcl]]} {
	source [file join $dir .. lib ttrace.tcl]
    } elseif {[file readable [file join $dir ttrace.tcl]]} {
	source [file join $dir ttrace.tcl]
    } elseif {[file exists //zipfs:/lib/thread/ttrace.tcl]
	    || ![catch {zipfs mount [file join $dir libtcl9thread3.0.4.so] //zipfs:/lib/thread}]} {
	source //zipfs:/lib/thread/ttrace.tcl
    } elseif {[file exists //zipfs:/app/thread_library/ttrace.tcl]
	    || ![catch {zipfs mount [file join $dir libtcl9thread3.0.4.so] //zipfs:/app/thread_library}]} {
	source //zipfs:/app/thread_library/ttrace.tcl
    }
    if {[namespace which ::ttrace::update] ne ""} {
	::ttrace::update
    }
}} $dir]
package ifneeded Ttrace 3.0.4 \
    [list package require -exact ttrace 3.0.4]
