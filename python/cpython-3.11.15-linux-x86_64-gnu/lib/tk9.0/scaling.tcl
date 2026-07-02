# scaling.tcl --
#
# Contains scaling-related utility procedures.
#
# Copyright © 2022 Csaba Nemethi <csaba.nemethi@t-online.de>

# ::tk::ScalingPct --
#
# Returns the display's "scaling percentage" (the display resolution expressed
# as a percentage of 96dpi), rounded to the nearest multiple of 25 that is at
# least 100.
#
# On X11 systems (but not on SDL systems that claim to be X11), the first call
# of the command also sets [tk scaling] and ::tk::fontScalingFactor to values
# extracted from the X11 configuration.
#
# The command is called during Tk initialization, from icons.tcl, when the
# latter is sourced by tk.tcl.

proc ::tk::ScalingPct {} {
    set pct [expr {[tk scaling] * 75}]

    variable doneScalingInitX11
    if {![info exists doneScalingInitX11]} {
	set pct [::tk::ScalingInitX11 $pct]
	set doneScalingInitX11 1
    }

    #
    # Save the value of pct rounded to the nearest multiple
    # of 25 that is at least 100, in the variable scalingPct.
    # See "man n tk_scalingPct" for use of ::tk::scalingPct.
    #
    variable scalingPct
    for {set scalingPct 100} {1} {incr scalingPct 25} {
	if {$pct < $scalingPct + 12.5} {
	    break
	}
    }

    return $scalingPct
}

proc ::tk::ScalingInitX11 {pct} {
    set onX11 [expr {[tk windowingsystem] eq "x11"}]
    set usingSDL [expr {[info exists ::tk::sdltk] && $::tk::sdltk}]

    if {$onX11 && !$usingSDL} {
	set origPct $pct

	#
	# Try to get the window scaling factor (1 or 2), partly
	# based on https://wiki.archlinux.org/title/HiDPI
	#
	set winScalingFactor 1
	variable fontScalingFactor 1		;# needed in the file ttk/fonts
	if {[catch {exec ps -e | grep xfce4-session}] == 0} {		;# Xfce
	    if {[catch {exec xfconf-query -c xsettings \
		 -p /Gdk/WindowScalingFactor} result] == 0} {
		set winScalingFactor $result
		if {$winScalingFactor >= 2} {
		    set fontScalingFactor 2
		}
	    }

	    #
	    # The DPI value can be set in the "Fonts" tab of the "Appearance"
	    # dialog or (on Linux Lite 5+) via the "HiDPI Settings" dialog.
	    #
	} elseif {[catch {exec ps -e | grep mate-session}] == 0} {	;# MATE
	    if {[catch {exec gsettings get org.mate.interface \
		 window-scaling-factor} result] == 0} {
		if {$result == 0} {			;# means: "Auto-detect"
		    #
		    # Try to get winScalingFactor from the cursor size
		    #
		    if {[catch {exec xrdb -query | grep Xcursor.size} result]
			== 0 &&
			[catch {exec gsettings get org.mate.peripherals-mouse \
			 cursor-size} defCursorSize] == 0} {
			set cursorSize [lindex $result 1]
			set winScalingFactor \
			    [expr {($cursorSize + $defCursorSize - 1) /
				   $defCursorSize}]
		    }
		} else {
		    set winScalingFactor $result
		}
	    }

	    #
	    # The DPI value can be set via the "Font Rendering Details"
	    # dialog, which can be opened using the "Details..." button
	    # in the "Fonts" tab of the "Appearance Preferences" dialog.
	    #
	} elseif {[catch {exec ps -e | grep gnome-session}] == 0 &&
		  [catch {exec gsettings get \
		   org.gnome.settings-daemon.plugins.xsettings overrides} \
		   result] == 0 &&
		  [set idx \
		   [string first "'Gdk/WindowScalingFactor'" $result]] >= 0} {
	    scan [string range $result $idx end] "%*s <%d>" winScalingFactor
	}

	#
	# Get the scaling percentage
	#
	if {$winScalingFactor >= 2} {
	    set pct 200
	} elseif {[catch {exec xrdb -query | grep Xft.dpi} result] == 0} {
	    #
	    # Derive the value of pct from that of the font DPI
	    #
	    set dpi [lindex $result 1]
	    set pct [expr {100.0 * $dpi / 96}]
	} elseif {[catch {exec ps -e | grep gnome-session}] == 0 &&
		  ![info exists ::env(WAYLAND_DISPLAY)] &&
		  [catch {exec xrandr | grep " connected"} result] == 0 &&
		  [catch {open $::env(HOME)/.config/monitors.xml} chan] == 0} {
	    #
	    # Update pct by scanning the file ~/.config/monitors.xml
	    #
	    ScanMonitorsFile $result $chan pct
	}

	if {($pct != 100) && ($pct != $origPct) && (![interp issafe])} {
	    #
	    # Set Tk's scaling factor according to $pct
	    #
	    tk scaling [expr {$pct / 75.0}]
	}
    }
    return $pct
}

# ::tk::ScaleNum --
#
# Scales an integer according to the display's current scaling percentage.
#
# Arguments:
#   num - An integer.

proc ::tk::ScaleNum num {
    return [expr {round($num * [tk scaling] * 0.75)}]
}

# ::tk::FontScalingFactor --
#
# Accessor command for variable ::tk::fontScalingFactor.

proc ::tk::FontScalingFactor {} {
    variable fontScalingFactor
    if {[info exists fontScalingFactor]} {
	return $fontScalingFactor
    } else {
	return 1
    }
}

# ::tk::ScanMonitorsFile --
#
# Updates the scaling percentage by scanning the file ~/.config/monitors.xml.
#
# Arguments:
#   xrandrResult - The output of 'xrandr | grep " connected"'.
#   chan -	   Returned from 'open ~/.config/monitors.xml'.
#   pctName -	   The name of a variable containing the scaling percentage.

proc ::tk::ScanMonitorsFile {xrandrResult chan pctName} {
    upvar $pctName pct

    #
    # Get the list of connected outputs reported by xrandr
    #
    set outputList {}
    foreach line [split $xrandrResult "\n"] {
	set idx [string first " " $line]
	set output [string range $line 0 [incr idx -1]]
	lappend outputList $output
    }
    set outputList [lsort $outputList]

    #
    # Get the content of the file ~/.config/monitors.xml
    #
    set str [read $chan]
    close $chan

    #
    # Run over the file's "configuration" sections
    #
    set idx 0
    while {[set idx2 [string first "<configuration>" $str $idx]] >= 0} {
	set idx2 [string first ">" $str $idx2]
	set idx [string first "</configuration>" $str $idx2]
	set config [string range $str [incr idx2] [incr idx -1]]

	#
	# Get the list of connectors within this configuration
	#
	set connectorList {}
	foreach {dummy connector} [regexp -all -inline \
		{<connector>([^<]+)</connector>} $config] {
	    lappend connectorList $connector
	}
	set connectorList [lsort $connectorList]

	#
	# If $outputList and $connectorList are identical then set the
	# variable pct to 100, 200, 300, 400, or 500, depending on the
	# max. scaling within this configuration, and exit the loop
	#
	if {$outputList eq $connectorList} {
	    set maxScaling 1.0
	    foreach {dummy scaling} [regexp -all -inline \
		    {<scale>([^<]+)</scale>} $config] {
		if {$scaling > $maxScaling} {
		    set maxScaling $scaling
		}
	    }

	    foreach n {4 3 2 1 0} {
		if {$maxScaling > $n} {
		    set pct [expr {($n + 1) * 100}]
		    break
		}
	    }

	    break
	}
    }
}
