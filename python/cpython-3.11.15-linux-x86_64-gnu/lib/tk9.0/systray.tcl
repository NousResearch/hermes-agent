# systray.tcl --

# This file defines the 'tk systray' command for icon display and manipulation
# in the system tray on X11, Windows, and macOS, and the 'tk sysnotify' command
# for system alerts on each platform. It implements an abstraction layer that
# presents a consistent API across the three platforms.

# Copyright © 2020 Kevin Walzer/WordTech Communications LLC.
# Copyright © 2020 Eric Boudaillier.
# Copyright © 2020 Francois Vogel.
#
# See the file "license.terms" for information on usage and redistribution
# of this file, and for a DISCLAIMER OF ALL WARRANTIES.

# Pure-Tcl system tooltip window for use with system tray icon if native
# implementation not available.

namespace eval ::tk::systray {
    variable _created 0
    variable _options {-image "" -text "" -button1 "" -button3 ""}
    variable _current {}
    variable _ico

    proc _balloon {w help} {
	bind $w <Any-Enter> "after 100 [list [namespace current]::_balloon_show %W [list $help] cursor]"
	bind $w <Any-Leave> "destroy %W._balloon"
    }

    proc _balloon_show {w msg i} {
	if {![winfo exists $w]} { return }

	# Use string match to allow that the help will be shown when
	# the pointer is in any child of the desired widget
	if {([winfo class $w] ne "Menu") && ![string match $w* [eval [list winfo containing] \
								    [winfo pointerxy $w]]]} {
	    return
	}

	set top $w._balloon
	::destroy $top
	toplevel $top -bg black -bd 1
	wm overrideredirect $top 1
	wm state $top withdrawn
	if {[tk windowingsystem] eq "aqua"}  {
	    ::tk::unsupported::MacWindowStyle style $top help none
	}
	pack [message $top._txt -aspect 10000 -text $msg]

	update idletasks
	set screenw [winfo screenwidth $w]
	set screenh [winfo screenheight $w]
	set reqw [winfo reqwidth $top]
	set reqh [winfo reqheight $top]
	# When adjusting for being on the screen boundary, check that we are
	# near the "edge" already, as Tk handles multiple monitors oddly
	if {$i eq "cursor"} {
	    set y [expr {[winfo pointery $w]+20}]
	    if {($y < $screenh) && ($y+$reqh) > $screenh} {
		set y [expr {[winfo pointery $w]-$reqh-5}]
	    }
	} elseif {$i ne ""} {
	    set y [expr {[winfo rooty $w]+[winfo vrooty $w]+[$w yposition $i]+25}]
	    if {($y < $screenh) && ($y+$reqh) > $screenh} {
		# show above if we would be offscreen
		set y [expr {[winfo rooty $w]+[$w yposition $i]-$reqh-5}]
	    }
	} else {
	    set y [expr {[winfo rooty $w]+[winfo vrooty $w]+[winfo height $w]+5}]
	    if {($y < $screenh) && ($y+$reqh) > $screenh} {
		# show above if we would be offscreen
		set y [expr {[winfo rooty $w]-$reqh-5}]
	    }
	}
	if {$i eq "cursor"} {
	    set x [winfo pointerx $w]
	} else {
	    set x [expr {[winfo rootx $w]+[winfo vrootx $w]+ ([winfo width $w]-$reqw)/2}]
	}
	# only readjust when we would appear right on the screen edge
	if {$x<0 && ($x+$reqw)>0} {
	    set x 0
	} elseif {($x < $screenw) && ($x+$reqw) > $screenw} {
	    set x [expr {$screenw-$reqw}]
	}
	if {[tk windowingsystem] eq "aqua"} {
	    set focus [focus]
	}

	wm geometry $top +$x+$y
	wm deiconify $top
	raise $top

	if {[tk windowingsystem] eq "aqua" && $focus ne ""} {
	    # Aqua's help window steals focus on display
	    after idle [list focus -force $focus]
	}
    }

    proc _win_callback {msg} {
	variable _current
	# The API at the Tk level does not feature bindings to double  clicks. Whatever
	# the speed the user clicks with, he expects the single click binding to fire.
	# Therefore we need to bind to both WM_*BUTTONDOWN and to WM_*BUTTONDBLCLK.
	switch -exact -- $msg {
	    WM_LBUTTONDOWN - WM_LBUTTONDBLCLK {
		uplevel #0 [dict get $_current -button1]
	    }
	    WM_RBUTTONDOWN - WM_RBUTTONDBLCLK {
		uplevel #0 [dict get $_current -button3]
	    }
	}
    }

    namespace export create configure destroy exists
    namespace ensemble create
}


# Pure-Tcl system notification window for use if native implementation not available.
# This is supposed to happen only on X11 when libnotify is not present.
namespace eval ::tk::sysnotify:: {
    # These defaults mimic the default behaviour of gnome and xfce notifications.
    # These are hardcoded defaults.
    variable defaults {
	padX            3
	padY            3
	background      gray15
	foreground      white
	delay           10000
	alpha           0.85
    }
    # These options are meant to be "public". The user could tinker with
    # these values to adjust the system notification appearance/behaviour.
    option add *Sysnotify.padX       [dict get $defaults padX]
    option add *Sysnotify.padY       [dict get $defaults padY]
    option add *Sysnotify.background [dict get $defaults background]
    option add *Sysnotify.foreground [dict get $defaults foreground]
    option add *Sysnotify.delay      [dict get $defaults delay]
    option add *Sysnotify.alpha      [dict get $defaults alpha]

    proc _notifywindow {title msg} {
	variable defaults

	# cleanup any previous notify window and create a new one
	set w ._notify
	_notifyDestroy $w
	toplevel $w -class Sysnotify

	# read the option database to check out whether the user has set
	# some options; fall back to our hardcoded defaults otherwise
	dict for {option value} [dict remove $defaults alpha] {
	    set $option [option get $w $option ""]
	    if {[set $option] eq ""} {
		set $option $value
	    }
	}

	set xpos [tk::ScaleNum 16]
	set ypos [tk::ScaleNum 48]
	# position from the "ne" corner
	wm geometry $w -$xpos+$ypos
	wm overrideredirect $w true

	# internal options
	option add *Sysnotify.Label.anchor     w
	option add *Sysnotify.Label.justify    left
	option add *Sysnotify.Label.wrapLength [expr {[winfo screenwidth .] / 4}]
	foreach option {padX padY foreground background} {
	    option add *Sysnotify.Label.$option [set $option]
	}
	set icon ::tk::icons::information
	set width [expr {[image width $icon] + 2 * $padX}]
	set height [expr {[image height $icon] + 2 * $padY}]
	label $w.icon -image $icon -width $width -height $height -anchor c
	label $w.title -text $title -font TkHeadingFont
	label $w.message -text [_filterMarkup $msg] -font TkTooltipFont

	grid $w.icon $w.title -sticky news
	grid ^       $w.message -sticky news

	bind Sysnotify <Map> [namespace code {
	    # set the wm attribute here; it is ignored if set
	    # before the window is mapped
	    wm attributes %W -alpha 0.0
	    if {[wm attributes %W -alpha] == 0.0} {
		_fadeIn %W
	    }
	}]
	bind Sysnotify <Enter> [namespace code {_onEnter %W}]
	bind Sysnotify <Leave> [namespace code {_onLeave %W}]
	bind $w <Button-1> [namespace code [list _notifyDestroy $w]]
	after $delay [namespace code [list _fadeOut $w]]
	return
    }

    # Fade the window into view.
    proc _fadeIn {w} {
	variable defaults
	if {![winfo exists $w]} {return}
	if {[set alpha  [option get $w alpha ""]] eq ""} {
	    set alpha [dict get $defaults alpha]
	}
	raise $w
	set before [wm attributes $w -alpha]
	set new    [expr { min($alpha, $before + 0.10) }]
	wm attributes $w -alpha $new
	set after  [wm attributes $w -alpha]
	if {($before == 1.0) || ($before == $after)} {
	    # not supported or we're done
	    return
	}
	after 40 [namespace code [list _fadeIn $w]]
    }

    # Fade out and destroy window.
    proc _fadeOut {w} {
	if {![winfo exists $w]} {return}
	set before [wm attributes $w -alpha]
	set new    [expr { $before - 0.02 }]
	wm attributes $w -alpha $new
	set after  [wm attributes $w -alpha]
	if {($after == 1.0) || ($before == $after)} {
	    _notifyDestroy $w
	    return
	}
	after 40 [namespace code [list _fadeOut $w]]
    }

    proc _notifyDestroy {w} {
	# cancel any pending fade in or fade out
	_cancelFading $w
	destroy $w
    }

    proc _onEnter {w} {
	wm attributes $w -alpha 1.0
	_cancelFading $w
    }

    proc _onLeave {w} {
	variable defaults
	if {[set alpha [option get $w alpha ""]] eq ""} {
	    set alpha [dict get $defaults alpha]
	}
	if {[set delay [option get $w delay ""]] eq ""} {
	    set delay [dict get $defaults delay]
	}
	wm attributes $w -alpha $alpha
	after $delay [namespace code [list _fadeOut $w]]
    }

    proc _cancelFading {w} {
	after cancel [namespace code [list _fadeOut $w]]
	after cancel [namespace code [list _fadeIn $w]]
    }

    # The Desktop Notifications Specification allow for some markup
    # in the message to display. It also specifies
    # "Notification servers that do not support these tags should
    # filter them out"
    # See https://specifications.freedesktop.org/notification-spec/latest/ar01s04.html
    # We don't event try to render those properly
    proc _filterMarkup {txt} {
	# remove fixed tags
	set maplist {<b> "" </b> "" <i> "" </i> "" <u> "" </u> "" </a> ""}
	set txt [string map $maplist $txt]
	# remove <img> tags leaving (possible) alt text
	set txt [regsub -- {<img *src="[^"]*" *(alt="([^"]*)")? */?>} $txt {\2}]
	# remove <a href=""> variable tag
	set txt [regsub -- {<a[^>]*>} $txt {}]
	return $txt
    }
}


# tk systray --
# This procedure creates an icon display in the platform-specific system tray.
#
# Subcommands:
#
#     create - create systray icon.
#         Arguments:
#             -image - Tk image to display.
#             -text - string to display in tooltip over image.
#             -button1 - Tcl proc to invoke on <Button-1> event.
#             -button3 - Tcl proc to invoke on <Button-3> event.
#
#     configure - change one of the systray properties.
#         Arguments (Any or all can be called):
#             -image - Tk image to update.
#             -text - string to update.
#             -button1 - Tcl proc to change for <Button-1> event.
#             -button3  - Tcl proc to change for <Button-3> event.
#
#     destroy - destroy systray icon.
#         Arguments:
#             none.
proc ::tk::systray::create {args} {
    variable _created
    variable _options
    variable _current
    variable _ico

    if {$_created} {
	return -code error -errorcode {TK SYSTRAY CREATE} "only one system tray icon supported per interpeter"
    }
    _check_options $args 0
    if {![dict exists $args -image]} {
	return -code error -errorcode {TK SYSTRAY CREATE} "missing required option \"-image\""
    }
    set values [dict merge $_options $args]
    try {
	switch -- [tk windowingsystem] {
	    "win32" {
		set _ico [_systray add -image [dict get $values -image] \
			-text [dict get $values -text] \
			-callback [list ::tk::systray::_win_callback %m]]
	    }
	    "x11" {
		_systray ._tray -image [dict get $values -image] -visible true
		_balloon ._tray [dict get $values -text]
		bind ._tray <Button-1> [dict get $values -button1]
		bind ._tray <Button-3> [dict get $values -button3]
	    }
	    "aqua" {
		_systray create [dict get $values -image] [dict get $values -text] \
			[dict get $values -button1] [dict get $values -button3]
	    }
	}
    } on ok {} {
	set _current $values
	set _created 1
	return
    } on error {msg opts} {
	return -code error -errorcode [dict get $opts -errorcode] $msg
    }
}

# Modify the systray icon.
proc ::tk::systray::configure {args} {
    variable _created
    variable _options
    variable _current
    variable _ico

    if {!$_created} {
	return -code error -errorcode {TK SYSTRAY CREATE} "systray not created"
    }
    _check_options $args 1
    if {[llength $args] == 0} {
	return $_current
    } elseif {[llength $args] == 1} {
	return [dict get $_current [lindex $args 0]]
    }
    set values [dict merge $_current $args]
    try {
	switch -- [tk windowingsystem] {
	    "win32" {
		if {[dict exists $args -image]} {
		    _systray modify $_ico -image [dict get $args -image]
		}
		if {[dict exists $args -text]} {
		    _systray modify $_ico -text [dict get $args -text]
		}
	    }
	    "x11" {
		if {[dict exists $args -image]} {
		    ._tray configure -image [dict get $args -image]
		}
		if {[dict exists $args -text]} {
		    _balloon ._tray [dict get $args -text]
		}
		if {[dict exists $args -button1]} {
		    bind ._tray <Button-1> [dict get $args -button1]
		}
		if {[dict exists $args -button3]} {
		    bind ._tray <Button-3> [dict get $args -button3]
		}
	    }
	    "aqua" {
		foreach {key opt} {image -image text \
			-text b1_callback -button1 b3_callback -button3} {
		    if {[dict exists $args $opt]} {
			_systray modify $key [dict get $args $opt]
		    }
		}
	    }
	}
    } on ok {} {
	set _current $values
	return
    } on error {msg opts} {
	return -code error -errorcode [dict get $opts -errorcode] $msg
    }
}

# Remove the systray icon.
proc ::tk::systray::destroy {} {
    variable _created
    variable _current
    variable _ico

    if {!$_created} {
	return -code error -errorcode {TK SYSTRAY DESTROY} "systray not created"
    }
    switch -- [tk windowingsystem] {
	"win32" {
	    _systray delete $_ico
	    set _ico ""
	}
	"x11" {
	    ::destroy ._tray
	}
	"aqua" {
	    _systray destroy
	}
    }
    set _created 0
    set _current {}
    return
}

# Check systray icon existence.
proc tk::systray::exists {} {
    variable _created
    return $_created
}

# Check systray options
proc ::tk::systray::_check_options {argsList singleOk} {
    variable _options

    set len [llength $argsList]
    while {[llength $argsList] > 0} {
	set opt [lindex $argsList 0]
	if {![dict exists $_options $opt]} {
	    tailcall return -code error -errorcode {TK SYSTRAY OPTION} \
		"unknown option \"$opt\": must be -image, -text, -button1 or -button3"
	}
	if {[llength $argsList] == 1 && !($len == 1 && $singleOk)} {
	    tailcall return -code error -errorcode {TK SYSTRAY OPTION} \
		"missing value for option \"$opt\""
	}
	set argsList [lrange $argsList 2 end]
    }
}

# tk sysnotify --
# This procedure implements a platform-specific system notification alert.
#
#   Arguments:
#       title - main text of alert.
#       message - body text of alert.

proc ::tk::sysnotify::sysnotify {title message} {

    switch -- [tk windowingsystem] {
	"win32" {
	    if {!$::tk::systray::_created} {
		error "must create a system tray icon with the \"tk systray\" command first"
	    }
	    _sysnotify notify $::tk::systray::_ico $title $message
	}
	"x11" {
	    if {[info commands ::tk::sysnotify::_sysnotify] eq ""} {
		_notifywindow $title $message
	    } else {
		_sysnotify $title $message
	    }
	}
	"aqua" {
	    _sysnotify $title $message
	}
    }
    return
}

#Add these commands to the tk command ensemble: tk systray, tk sysnotify
#Thanks to Christian Gollwitzer for the guidance here
namespace ensemble configure tk -map \
    [dict merge [namespace ensemble configure tk -map] \
	{systray ::tk::systray sysnotify ::tk::sysnotify::sysnotify}]

