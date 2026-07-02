# print.tcl --

# This file defines the 'tk print' command for printing of the canvas
# widget and text on X11, Windows, and macOS. It implements an abstraction
# layer that presents a consistent API across the three platforms.

# Copyright © 2009 Michael I. Schwartz.
# Copyright © 2021 Kevin Walzer/WordTech Communications LLC.
# Copyright © 2021 Harald Oehlmann, Elmicron GmbH
# Copyright © 2022 Emiliano Gavilan
#
# See the file "license.terms" for information on usage and redistribution
# of this file, and for a DISCLAIMER OF ALL WARRANTIES.

namespace eval ::tk::print {
    namespace import -force ::tk::msgcat::*

    # margins for printing text, in mm
    variable margin
    set margin(top)    15
    set margin(left)   25
    set margin(right)  15
    set margin(bottom) 15

    # makeTempFile:
    #    Create a temporary file and populate its contents
    # Arguments:
    #	 filename - base of the name of the file to create
    #    contents - what to put in the file; defaults to empty
    # Returns:
    #    Full filename for created file
    #
    proc makeTempFile {filename {contents ""}} {
	set dumpfile [file join /tmp rawprint.txt]
	set tmpfile [file join /tmp $filename]
	set f [open $dumpfile w]
	try {
	    puts -nonewline $f $contents
	} finally {
	    close $f
	    if {[file extension $filename] == ".ps"} {
		#don't apply formatting to PostScript
		file rename -force $dumpfile $tmpfile
	    } else {
	    #Make text fixed width for improved printed output
		exec fmt -w 75 $dumpfile > $tmpfile
	    }
	    return $tmpfile
	}
    }

    # _wrapLines -
    #   wrap long lines into lines of at most length wl at word boundaries
    # Arguments:
    #   str   - string to be wrapped
    #   wl    - wrap length
    #
    proc _wrapLines {str wl} {
	# This is a really simple algorithm: it breaks a line on space or tab
	# character, collapsing them only at the breaking point.
	# Leading space is left as-is.
	# For a full fledged line breaking algorithm see
	# Unicode® Standard Annex #14 "Unicode Line Breaking Algorithm"
	set res {}
	incr wl -1
	set re [format {((?:^|[^[:blank:]]).{0,%d})(?:[[:blank:]]|$)} $wl]
	foreach line [split $str \n] {
	    lappend res {*}[lmap {_ l} [regexp -all -inline -- $re $line] {
		set l
	    }]
	}
	# the return value is a list of lines
	return $res
    }

    if {[tk windowingsystem] eq "win32"} {
	variable printer_name
	variable copies
	variable dpi_x
	variable dpi_y
	variable paper_width
	variable paper_height
	variable margin_left
	variable margin_top
	variable printargs
	array set printargs {}

	# Multiple utility procedures for printing text based on the
	# C printer primitives.

	# _set_dc:
	# Select printer and set device context and other parameters
	# for print job.
	#
	proc _set_dc {} {
	    variable margin
	    variable printargs
	    variable printer_name
	    variable paper_width
	    variable paper_height
	    variable dpi_x
	    variable dpi_y
	    variable copies

	    #First, we select the printer.
	    _selectprinter

	    #Next, set values. Some are taken from the printer,
	    #some are sane defaults.

	    if {[info exists printer_name]} {
		set printargs(hDC) $printer_name
		set printargs(pw) $paper_width
		set printargs(ph) $paper_height
		set printargs(resx) $dpi_x
		set printargs(resy) $dpi_y
		set printargs(copies) $copies
		set printargs(resolution) [list $dpi_x $dpi_y]
	    }
	}

	# _print
	# Main entry of the print subsystem for Win32.
	# Set the print job name and delegate to the right procedure
	proc _print {w} {
	    variable jobname

	    set class [winfo class $w]
	    if {$class ni {Text Canvas}} {
		return -code error "can not print widget of class \"$class\":\
		    should be Canvas or Text"
	    }
	    set jobname "[tk appname]: Tk widget $w"
	    switch -- $class {
		Text {
		    _print_data2 [$w get 1.0 end] {{Courier New} 11}
		}
		Canvas {
		    _print_widget $w
		}
	    }
	}

	# _print_data2
	# This function prints multiple-page files, using a line-oriented
	# function, taking advantage of knowing the character widths.
	# Arguments:
	# data -       Text data for printing
	# font -       Font for printing
	proc _print_data2 { data font } {
	    variable printargs
	    variable printer_name
	    variable jobname

	    _set_dc

	    if {![info exists printer_name]} {
		return
	    }

	    set maxwidth [expr {
		($printargs(pw) - $printargs(lm) - $printargs(rm)) *
		$printargs(resx) / 1000
	    }]
	    set maxheight [expr {
		($printargs(ph) - $printargs(tm) - $printargs(bm)) *
		$printargs(resy) / 1000
	    }]
	    set lm [expr {$printargs(lm) * $printargs(resx) / 1000}]
	    set tm [expr {$printargs(tm) * $printargs(resy) / 1000}]
	    set curheight $tm

	    lassign [_opendoc $jobname $font] charwidth charheight
	    _openpage

	    set wl [expr {($maxwidth / $charwidth) + 1}]
	    foreach line [_wrapLines $data $wl] {
		_gdi textplain $printargs(hDC) $lm $curheight $line
		incr curheight $charheight
		if {$curheight + $charheight > $maxheight} {
		    _closepage
		    _openpage
		    set curheight $tm
		}
	    }
	    _closepage
	    _closedoc
	    return
	}

	# _print_data
	# This function prints multiple-page files, using a line-oriented
	# function, taking advantage of knowing the character widths.
	# Arguments:
	# data -       Text data for printing
	# font -       Font for printing
	proc _print_data { data font } {
	    variable printargs
	    variable printer_name
	    variable charwidths

	    _set_dc

	    if {![info exists printer_name]} {
		return
	    }

	    _gdi characters $printargs(hDC) -font $font \
		-array ::tk::print::charwidths
	    array default set ::tk::print::charwidths $charwidths(x)
	    # check for a monospaced font
	    set mono 0
	    if {$charwidths(i) == $charwidths(m)} {
		set mono $charwidths(m)
	    }
	    set maxwidth [expr {
		($printargs(pw) - $printargs(lm) - $printargs(rm)) *
		$printargs(resx) / 1000
	    }]
	    set maxhgt [expr {
		($printargs(ph) - $printargs(tm) - $printargs(bm)) *
		$printargs(resy) / 1000
	    }]
	    set lm [expr {$printargs(lm) * $printargs(resx) / 1000}]
	    set tm [expr {$printargs(tm) * $printargs(resy) / 1000}]
	    set curhgt $tm

	    _opendoc "[tk appname]: Tk Print Job"
	    _openpage

	    # if a monospaced font is used, things are a bit easier
	    # handle this case here
	    if {$mono > 0} {
		# can use the same line breaking algorithm as X11
		set wl [expr {($maxwidth / $mono) + 1}]
		foreach line [_wrapLines $data $wl] {
		    set hgt [_gdi text $printargs(hDC) $lm $curhgt \
			-anchor nw -justify left \
			-text $line -font $font]
		    incr curhgt $hgt
#		    puts "height: $hgt"
		    if {$curhgt + $hgt > $maxhgt} {
			_closepage
			_openpage
			set curhgt $tm
		    }
		}
		_closepage
		_closedoc
		return
	    }

	    # proportional font
	    set totallen [string length $data]
	    set curlen 0
	    set breaklines 1
	    while {$curlen < $totallen} {
		set linestring [string range $data $curlen end]
		if {$breaklines} {
		    set endind [string first "\n" $linestring]
		    if {$endind >= 0} {
			set linestring [string range $linestring 0 $endind]
			# handle blank lines....
			if {$linestring eq ""} {
			    set linestring " "
			}
		    }
		}
		lassign [_print_page_nextline $linestring $curhgt $font \
		    $lm $maxwidth] len hgt
		incr curlen $len
		incr curhgt $hgt
		if {$curhgt + $hgt > $maxhgt} {
		    _closepage
		    _openpage
		    set curhgt $zerohgt
		}
	    }
	    _closepage
	    _closedoc
	}

	# _print_page_nextline
	# Returns the pair "chars y"
	# where chars is the number of characters printed on the line
	# and y is the height of the line printed
	# Arguments:
	#   string -         Data to print
	#   y -              Y value to begin printing at, in logical units
	#   font -           if non-empty specifies a font to draw the line in
	#   lm -             left margin, in logical units
	#   maxwidth -       line length limit in logical units
	proc _print_page_nextline {string y font lm maxwidth} {
	    variable charwidths
	    variable printargs

	    set endindex 0
	    set totwidth 0
	    set maxstring [string length $string]

	    for {set i 0} {($i < $maxstring) && ($totwidth < $maxwidth)} {incr i} {
		incr totwidth $charwidths([string index $string $i])
	    }

	    set endindex $i
	    set startindex $endindex

	    if {$i < $maxstring} {
		# In this case, the whole data string is not used up, and we
		# wish to break on a word. Since we have all the partial
		# widths calculated, this should be easy.

		set endindex [expr {[string wordstart $string $endindex] - 1}]
		set startindex [expr {$endindex + 1}]

		# If the line is just too long (no word breaks), print as much
		# as you can....
		if {$endindex <= 1} {
		    set endindex $i
		    set startindex $i
		}
	    }

	    set txt [string trimright [string range $string 0 $endindex] "\r\n"]
	    if {$font ne ""} {
		set height [_gdi text $printargs(hDC) $lm $y \
				 -anchor nw -justify left \
				 -text $txt -font $font]
	    } else {
		set height [_gdi text $printargs(hDC) $lm $y \
				 -anchor nw -justify left -text $txt]
	    }
	    return [list $startindex $height]
	}

	# These procedures read in the canvas widget, and write all of
	# its contents out to the Windows printer.

	proc _init_print {} {
	    variable margin
	    variable printargs
	    variable vtgPrint

	    set vtgPrint(printer.bg) white

	    # convert margins to windows DC units (1000 point per inch)
	    set printargs(tm) [expr {int($margin(top) / 25.4 * 1000)}]
	    set printargs(lm) [expr {int($margin(left) / 25.4 * 1000)}]
	    set printargs(rm) [expr {int($margin(right) / 25.4 * 1000)}]
	    set printargs(bm) [expr {int($margin(bottom) / 25.4 * 1000)}]
	}

	# _print_widget
	# Main procedure for printing a widget.  Currently supports
	# canvas widgets.  Handles opening and closing of printer.
	# Arguments:
	#   wid -              The widget to be printed.
	#   name  -            App name to pass to printer.

	proc _print_widget {w} {
	    variable printargs
	    variable printer_name
	    variable jobname

	    # provide an early exit if the widget is not a canvas
	    set class [winfo class $w]
	    if {$class ne "Canvas"} {
		return -code error "Can't print items of type $class.\
		     No handler registered"
	    }

	    _set_dc

	    if {![info exists printer_name]} {
		return
	    }

	    _opendoc $jobname
	    _openpage

	    # Here is where any scaling/gdi mapping should take place
	    # For now, scale so the dimensions of the window are sized to the
	    # width of the page. Scale evenly.

	    # For normal windows, this may be fine--but for a canvas, one
	    # wants the canvas dimensions, and not the WINDOW dimensions.
	    if {$class eq "Canvas"} {
		set sc [$w cget -scrollregion]
		# if there is no scrollregion, use width and height.
		if {$sc eq ""} {
		    set window_x [winfo pixels $w [$w cget -width]]
		    set window_y [winfo pixels $w [$w cget -height]]
		} else {
		    set window_x [winfo pixels $w [lindex $sc 2]]
		    set window_y [winfo pixels $w [lindex $sc 3]]
		}
	    } else {
		set window_x [winfo width $w]
		set window_y [winfo height $w]
	    }

	    set printer_x [expr {
		( $printargs(pw) - $printargs(lm) - $printargs(rm) ) *
		$printargs(resx)  / 1000.0
	    }]
	    set printer_y [expr {
		( $printargs(ph) - $printargs(tm) - $printargs(bm) ) *
		$printargs(resy) / 1000.0
	    }]
	    set factor_x [expr {$window_x / $printer_x}]
	    set factor_y [expr {$window_y / $printer_y}]

	    if {$factor_x < $factor_y} {
		set lo $window_y
		set ph $printer_y
	    } else {
		set lo $window_x
		set ph $printer_x
	    }

	    _gdi map $printargs(hDC) -logical $lo -physical $ph \
		-offset $printargs(resolution)

	    # Handling of canvas widgets.
	    _print_canvas $printargs(hDC) $w

	    # End printing process.
	    _closepage
	    _closedoc
	}

	#  _print_canvas
	# Main procedure for writing canvas widget items to printer.
	# Arguments:
	#    hdc -              The printer handle.
	#    cw  -              The canvas widget.
	proc _print_canvas {hdc cw} {
	    variable vtgPrint
	    variable printargs

	    # Get information about page being printed to
	    # print_canvas.CalcSizing $cw
	    set vtgPrint(canvas.bg) [string tolower [$cw cget -background]]

	    # Re-write each widget from cw to printer
	    foreach id [$cw find all] {
		if {[$cw itemcget $id -state] eq "hidden"} {
		    # don't display hidden items
		    continue
		}
		set type [$cw type $id]
		if {[info commands _print_canvas.$type] eq "_print_canvas.$type"} {
		    _print_canvas.$type $printargs(hDC) $cw $id
		} else {
		    # should we use puts?
		    # puts "Omitting canvas item of type $type since there is no handler registered for it"
		}
	    }
	}

	# These procedures support the various canvas item types, reading the
	# information about the item on the real canvas and then writing a
	# similar item to the printer.

	# _print_canvas.line
	# Description:
	#   Prints a line item.
	# Arguments:
	#   hdc -              The printer handle.
	#   cw  -              The canvas widget.
	#   id  -              The id of the canvas item.
	proc _print_canvas.line {hdc cw id} {
	    variable vtgPrint
	    variable printargs

	    set color [_print_canvas.TransColor [$cw itemcget $id -fill]]
	    if {[string match $vtgPrint(printer.bg) $color]} {
		return
	    }

	    set coords [$cw coords $id]
	    set wdth   [$cw itemcget $id -width]
	    set arrow  [$cw itemcget $id -arrow]
	    set arwshp [$cw itemcget $id -arrowshape]
	    set dash   [$cw itemcget $id -dash]
	    set smooth [$cw itemcget $id -smooth]
	    set splinesteps [$cw itemcget $id -splinesteps]
	    set capstyle    [$cw itemcget $id -capstyle]
	    set joinstyle   [$cw itemcget $id -joinstyle]

	    set cmdargs {}

	    if {$wdth > 1} {
		lappend cmdargs -width $wdth
	    }
	    if {$dash ne ""} {
		lappend cmdargs -dash $dash
	    }
	    if {$smooth ne ""} {
		lappend cmdargs -smooth $smooth
	    }
	    if {$splinesteps ne ""} {
		lappend cmdargs -splinesteps $splinesteps
	    }

	    set result [_gdi line $hdc {*}$coords \
			    -fill $color -arrow $arrow -arrowshape $arwshp \
			    -capstyle $capstyle -joinstyle $joinstyle \
			    {*}$cmdargs]
	    if {$result ne ""} {
		puts $result
	    }
	}

	# _print_canvas.arc
	#   Prints a arc item.
	# Args:
	#   hdc -              The printer handle.
	#   cw  -              The canvas widget.
	#   id  -              The id of the canvas item.
	proc _print_canvas.arc {hdc cw id} {
	    variable vtgPrint
	    variable printargs

	    set color [_print_canvas.TransColor [$cw itemcget $id -outline]]
	    if {[string match $vtgPrint(printer.bg) $color]} {
		return
	    }
	    set coords  [$cw coords $id]
	    set wdth    [$cw itemcget $id -width]
	    set style   [$cw itemcget $id -style]
	    set start   [$cw itemcget $id -start]
	    set extent  [$cw itemcget $id -extent]
	    set fill    [$cw itemcget $id -fill]

	    set cmdargs {}
	    if {$wdth > 1} {
		lappend cmdargs -width $wdth
	    }
	    if {$fill ne ""} {
		lappend cmdargs -fill $fill
	    }

	    _gdi arc $hdc {*}$coords \
		-outline $color -style $style -start $start -extent $extent \
		{*}$cmdargs
	}

	# _print_canvas.polygon
	#   Prints a polygon item.
	# Arguments:
	#   hdc -              The printer handle.
	#   cw  -              The canvas widget.
	#   id  -              The id of the canvas item.
	proc _print_canvas.polygon {hdc cw id} {
	    variable vtgPrint
	    variable printargs

	    set fcolor [_print_canvas.TransColor [$cw itemcget $id -fill]]
	    if {$fcolor eq ""} {
		set fcolor $vtgPrint(printer.bg)
	    }
	    set ocolor [_print_canvas.TransColor [$cw itemcget $id -outline]]
	    if {$ocolor eq ""} {
		set ocolor $vtgPrint(printer.bg)
	    }
	    set coords  [$cw coords $id]
	    set wdth [$cw itemcget $id -width]
	    set smooth  [$cw itemcget $id -smooth]
	    set splinesteps [$cw itemcget $id -splinesteps]

	    set cmdargs {}
	    if {$smooth ne ""} {
		lappend cmdargs -smooth $smooth
	    }
	    if {$splinesteps ne ""} {
		lappend cmdargs -splinesteps $splinesteps
	    }

	    _gdi polygon $hdc {*}$coords \
		-width $wdth -fill $fcolor -outline $ocolor {*}$cmdargs
	}

	# _print_canvas.oval
	#   Prints an oval item.
	# Arguments:
	#   hdc -              The printer handle.
	#   cw  -              The canvas widget.
	#   id  -              The id of the canvas item.
	proc _print_canvas.oval {hdc cw id} {
	    variable vtgPrint

	    set fcolor [_print_canvas.TransColor [$cw itemcget $id -fill]]
	    if {$fcolor eq ""} {
		set fcolor $vtgPrint(printer.bg)
	    }
	    set ocolor [_print_canvas.TransColor [$cw itemcget $id -outline]]
	    if {$ocolor eq ""} {
		set ocolor $vtgPrint(printer.bg)
	    }
	    set coords  [$cw coords $id]
	    set wdth [$cw itemcget $id -width]

	    _gdi oval $hdc {*}$coords \
		-width $wdth -fill $fcolor -outline $ocolor
	}

	# _print_canvas.rectangle
	#   Prints a rectangle item.
	# Arguments:
	#   hdc -              The printer handle.
	#   cw  -              The canvas widget.
	#   id  -              The id of the canvas item.
	proc _print_canvas.rectangle {hdc cw id} {
	    variable vtgPrint

	    set fcolor [_print_canvas.TransColor [$cw itemcget $id -fill]]
	    if {$fcolor eq ""} {
		set fcolor $vtgPrint(printer.bg)
	    }
	    set ocolor [_print_canvas.TransColor [$cw itemcget $id -outline]]
	    if {$ocolor eq ""} {
		set ocolor $vtgPrint(printer.bg)
	    }
	    set coords  [$cw coords $id]
	    set wdth [$cw itemcget $id -width]

	    _gdi rectangle $hdc {*}$coords \
		-width $wdth -fill $fcolor -outline $ocolor
	}

	# _print_canvas.text
	#   Prints a text item.
	# Arguments:
	#   hdc -              The printer handle.
	#   cw  -              The canvas widget.
	#   id  -              The id of the canvas item.
	proc _print_canvas.text {hdc cw id} {
	    variable vtgPrint
	    variable printargs

	    set color [_print_canvas.TransColor [$cw itemcget $id -fill]]
#	    if {"white" eq [string tolower $color]} {
#		return
#	    }
	    # set color black
	    set txt [$cw itemcget $id -text]
	    if {$txt eq ""} {
		return
	    }
	    set coords [$cw coords $id]
	    set anchr [$cw itemcget $id -anchor]

	    set angle [$cw itemcget $id -angle]
	    set wdth [winfo pixels $cw [$cw itemcget $id -width]]
	    set just [$cw itemcget $id -justify]

	    # Get the real canvas font info suitable for printer
	    set font [_make_gdi_cfont $cw [$cw itemcget $id -font]]

	    _gdi text $hdc {*}$coords \
		-fill $color -text $txt -font $font \
		-anchor $anchr -width $wdth -justify $just -angle $angle
	}

	proc _make_gdi_cfont {w fname} {
	    set fa [font actual $fname]
	    # Transform to the third documented format in font(n)
	    # {family size style style style style}
	    set font [lmap k { -family -size -weight -slant } {
		dict get $fa $k
	    }]
	    # add underline and overstrike
	    foreach k { -underline -overstrike } v { underline overstrike } {
		if {[dict get $fa $k]} {
		    lappend font $v
		}
	    }
	    # make sure size is in pixels (negative)
	    if {[set size [lindex $font 1]] > 0} {
		lset font 1 [winfo pixels $w -${size}p]
	    }
	    return $font
	}

	# _print_canvas.image
	# Prints an image item.
	# Arguments:
	#   hdc -              The printer handle.
	#   cw  -              The canvas widget.
	#   id  -              The id of the canvas item.
	proc _print_canvas.image {hdc cw id} {
	    # First, we have to get the image name.
	    set imagename [$cw itemcget $id -image]
	    if {[image type $imagename] ne "photo"} {
		return
	    }

	    # Next, we get the location and anchor
	    lassign [$cw coords $id] ix iy
	    set anchor [$cw itemcget $id -anchor]

	    _gdi photo $hdc $ix $iy -anchor $anchor -photo $imagename
	}

	# _print_canvas.bitmap
	#   Prints a bitmap item.
	# Arguments:
	#   hdc -              The printer handle.
	#   cw  -              The canvas widget.
	#   id  -              The id of the canvas item.
	proc _print_canvas.bitmap {hdc cw id} {
	    # bitmap is not yet supported !!
	    # do nothing until it is (if ever).
	    return
	    variable option
	    variable vtgPrint

	    # First, we have to get the bitmap name.
	    set name [$cw itemcget $id -bitmap]

	    # Now we get the size.
#	    set wid [image width $imagename]
#	    set hgt [image height $imagename]
	    set bbox [$cw bbox $id]
	    set wid [expr {[lindex $bbox 2] - [lindex $bbox 0]}]
	    set hgt [expr {[lindex $bbox 3] - [lindex $bbox 1]}]

	    #Next, we get the location and anchor.
	    set anchor [$cw itemcget $id -anchor]
	    set coords [$cw coords $id]

	    # Since the GDI commands don't yet support images and bitmaps,
	    # and since this represents a rendered bitmap, we CAN use
	    # copybits IF we create a new temporary toplevel to hold the beast.
	    # If this is too ugly, change the option!

	    if {[info exist option(use_copybits)]} {
		set firstcase $option(use_copybits)
	    } else {
		set firstcase 0
	    }
	    if {$firstcase > 0} {
		set tl [toplevel .tmptop[expr {int( rand() * 65535 )}] \
			    -height $hgt -width $wid \
			    -background $vtgPrint(canvas.bg)]
		canvas $tl.canvas -width $wid -height $hgt
		$tl.canvas create bitmap 0 0 -bitmap $name -anchor nw
		pack $tl.canvas -side left -expand false -fill none
		tkwait visibility $tl.canvas
		update
		set srccoords [list 0 0 [expr {$wid - 1}] [expr {$hgt - 1}]]
		set dstcoords [list [lindex $coords 0] [lindex $coords 1] [expr {$wid - 1}] [expr {$hgt - 1}]]
		_gdi copybits $hdc -window $tl -client \
		    -source $srccoords -destination $dstcoords
		destroy $tl
	    } else {
		_gdi bitmap $hdc {*}$coords \
		    -anchor $anchor -bitmap $name
	    }
	}

	# These procedures transform attribute setting from the real
	# canvas to the appropriate setting for printing to paper.

	# _print_canvas.TransColor
	#   Does the actual transformation of colors from the
	#   canvas widget to paper.
	# Arguments:
	#   color -            The color value to be transformed.
	proc _print_canvas.TransColor {color} {
	    variable vtgPrint
	    variable printargs

	    switch [string toupper $color] {
		$vtgPrint(canvas.bg)       {return $vtgPrint(printer.bg)}
	    }
	    return $color
	}

	# Initialize all the variables once.
	_init_print
    }
    #end win32 procedures
}

# Begin X11 procedures. They depends on Cups being installed.
# X11 procedures abstracts print management with a "cups" ensemble command

#	cups defaultprinter	returns the default printer
#	cups getprinters	returns a dictionary of printers along
#				with printer info
#	cups print $printer $data ?$options?
#				print the data (binary) on a given printer
#				with the provided (supported) options:
#				-colormode -copies -format -margins
#				-media -nup -orientation
#				-prettyprint -title -tzoom

# Some output configuration that on other platforms is managed through
# the printer driver/dialog is configured through the canvas postscript command.
if {[tk windowingsystem] eq "x11"} {
    if {[info commands ::tk::print::cups] eq ""} {
	namespace eval ::tk::print::cups {
	    # Pure Tcl cups ensemble command implementation
	    variable pcache
	}

	proc ::tk::print::cups::defaultprinter {} {
	    set default {}
	    regexp {: ([^[:space:]]+)$} [exec lpstat -d] _ default
	    return $default
	}

	proc ::tk::print::cups::getprinters {} {
	    variable pcache
	    # Test for existence of lpstat command to obtain the list of
	    # printers.
	    # Return an error if not found.
	    set res {}
	    try {
		set printers [lsort -unique [split [exec lpstat -e] \n]]
		foreach printer $printers {
		    set options [Parseoptions [exec lpoptions -p $printer]]
		    dict set res $printer $options
		}
	    } trap {POSIX ENOENT} {e o} {
		# no such command in PATH
		set cmd [lindex [dict get $o -errorstack ] 1 2]
		return -code error "Unable to obtain the list of printers.\
		    Command \"$cmd\" not found.\
		    Please install the CUPS package for your system."
	    } trap {CHILDSTATUS} {e o} {
		# command returns a non-0 exit status. Wrong print system?
		set cmd [lindex [dict get $o -errorstack ] 1 2]
		return -code error "Command \"$cmd\" return with errors"
	    }
	    return [set pcache $res]
	}

	# Parseoptions
	#   Parse lpoptions -d output. It has three forms
	#     option-key
	#     option-key=option-value
	#     option-key='option value with spaces'
	# Arguments:
	#   data - data to process.
	#
	proc ::tk::print::cups::Parseoptions {data} {
	    set res {}
	    set re {[^ =]+|[^ ]+='[^']+'|[^ ]+=[^ ']+}
	    foreach tok [regexp -inline -all $re $data] {
		lassign [split $tok "="] k v
		dict set res $k [string trim $v "'"]
	    }
	    return $res
	}

	proc ::tk::print::cups::print {printer data args} {
	    variable pcache
	    if {$printer ni [dict keys $pcache]} {
		return -code error "unknown printer or class \"$printer\""
	    }
	    set title "Tk print job"
	    set options {
		-colormode -copies -format -margins -media -nup -orientation
		-prettyprint -title -tzoom
	    }
	    while {[llength $args]} {
		set opt [tcl::prefix match $options [lpop args 0]]
		switch  $opt {
		    -colormode {
			set opts {auto monochrome color}
			set val [tcl::prefix match $opts [lpop args 0]]
			lappend printargs -o print-color-mode=$val
		    }
		    -copies {
			set val [lpop args 0]
			if {![string is integer -strict $val] ||
			    $val < 0 || $val > 100
			} {
			    # save paper !!
			    return -code error "copies must be an integer\
				between 0 and 100"
			}
			lappend printargs -o copies=$val
		    }
		    -format {
			set opts {auto pdf postscript text}
			set val [tcl::prefix match $opts [lpop args 0]]
			# lpr uses auto always
		    }
		    -margins {
			set val [lpop args 0]
			if {[llength $val] != 4 ||
			    ![string is integer -strict [lindex $val 0]] ||
			    ![string is integer -strict [lindex $val 1]] ||
			    ![string is integer -strict [lindex $val 2]] ||
			    ![string is integer -strict [lindex $val 3]]
			} {
			    return -code error "margins must be a list of 4\
				integers: top left bottom right"
			}
			lappend printargs -o page-top=[lindex $val 0]
			lappend printargs -o page-left=[lindex $val 1]
			lappend printargs -o page-bottom=[lindex $val 2]
			lappend printargs -o page-right=[lindex $val 3]
		    }
		    -media {
			set opts {a4 legal letter}
			set val [tcl::prefix match $opts [lpop args 0]]
			lappend printargs -o media=$val
		    }
		    -nup {
			set val [lpop args 0]
			if {$val ni {1 2 4 6 9 16}} {
			    return -code error "number-up must be 1, 2, 4, 6, 9 or\
				16"
			}
			lappend printargs -o number-up=$val
		    }
		    -orientation {
			set opts {portrait landscape}
			set val [tcl::prefix match $opts [lpop args 0]]
			if {$val eq "landscape"}
			lappend printargs -o landscape=true
		    }
		    -prettyprint {
			lappend printargs -o prettyprint=true
			# prettyprint mess with these default values if set
			# so we force them.
			# these will be overriden if set after this point
			if {[lsearch $printargs {cpi=*}] == -1} {
			    lappend printargs -o cpi=10.0
			    lappend printargs -o lpi=6.0
			}
		    }
		    -title {
			set title [lpop args 0]
		    }
		    -tzoom {
			set val [lpop args 0]
			if {![string is double -strict $val] ||
			    $val < 0.5 || $val > 2.0
			 } {
			    return -code error "text zoom must be a number between\
				0.5 and 2.0"
			}
			# CUPS text filter defaults to lpi=6 and cpi=10
			lappend printargs -o cpi=[expr {10.0 / $val}]
			lappend printargs -o lpi=[expr {6.0 / $val}]
		    }
		    default {
			# shouldn't happen
		    }
		}
	    }
	    # build our options
	    lappend printargs -T $title
	    lappend printargs -P $printer
	    # open temp file
	    set fd [file tempfile fname tk_print]
	    chan configure $fd -translation binary
	    chan puts $fd $data
	    chan close $fd
	    # add -r to automatically delete temp files
	    exec lpr {*}$printargs -r $fname &
	}

	namespace eval ::tk::print::cups {
	    namespace export defaultprinter getprinters print
	    namespace ensemble create
	}
    };# ::tk::print::cups

    namespace eval ::tk::print {

	variable mcmap
	set mcmap(media) [dict create \
	    [mc "Letter"]	letter \
	    [mc "Legal"]	legal \
	    [mc "A4"]		a4]
	set mcmap(orient) [dict create \
	    [mc "Portrait"]	portrait \
	    [mc "Landscape"]	landscape]
	set mcmap(color) [dict create \
	    [mc "RGB"]		color \
	    [mc "Grayscale"]	gray]

	# available print options
	variable optlist
	set optlist(printer)	{}
	set optlist(media)	[dict keys $mcmap(media)]
	set optlist(orient)	[dict keys $mcmap(orient)]
	set optlist(color)	[dict keys $mcmap(color)]
	set optlist(number-up)	{1 2 4 6 9 16}

	# selected options
	variable option
	set option(printer)	{}
	# Initialize with sane defaults.
	set option(copies)	1
	set option(media)	[mc "A4"]
	# Canvas options
	set option(orient)	[mc "Portrait"]
	set option(color)	[mc "RGB"]
	set option(czoom)	100
	# Text options.
	# See libcupsfilter's cfFilterTextToPDF() and cups-filters's texttopdf
	# known options:
	# prettyprint, wrap, columns, lpi, cpi
	set option(number-up)	1
	set option(tzoom)	100; # we derive lpi and cpi from this value
	set option(pprint)	0  ; # pretty print
	# convert margins from mm to points
	variable margin
	set option(margin-top)    [expr {int($margin(top) * 72 / 25.4)}]
	set option(margin-left)   [expr {int($margin(left) * 72 / 25.4)}]
	set option(margin-right)  [expr {int($margin(right) * 72 / 25.4)}]
	set option(margin-bottom) [expr {int($margin(bottom) * 72 / 25.4)}]

	# array to collect printer information
	variable pinfo
	array set pinfo {}

	# a map for printer state -> human readable message
	variable statemap
	dict set statemap 3 [mc "Idle"]
	dict set statemap 4 [mc "Printing"]
	dict set statemap 5 [mc "Printer stopped"]
    }

    # ttk version of [tk_optionMenu]
    # var should be a full qualified varname
    proc ::tk::print::ttk_optionMenu {w var args} {
	ttk::menubutton $w -textvariable $var -menu $w.menu
	menu $w.menu
	foreach option $args {
	    $w.menu add command \
		-label $option \
		-command [list set $var $option]
	}
	# return the same value as tk_optionMenu
	return $w.menu
    }

    # _setprintenv
    #   Set the print environtment - list of printers, state and options.
    # Arguments:
    #   none.
    #
    proc ::tk::print::_setprintenv {} {
	variable option
	variable optlist
	variable pinfo

	set optlist(printer) {}
	dict for {printer options} [cups getprinters] {
	    lappend optlist(printer) $printer
	    set pinfo($printer) $options
	}

	# It's an error to not have any printer configured
	if {[llength $optlist(printer)] == 0} {
	    return -code error "No installed printers found.\
	    Please check or update your CUPS installation."
	}

	# If no printer is selected, check for the default one
	# If none found, use the first one from the list
	if {$option(printer) eq ""} {
	    set option(printer) [cups defaultprinter]
	    if {$option(printer) eq ""} {
		set option(printer) [lindex $optlist(printer) 0]
	    }
	}
    }

    # _print
    #   Main printer dialog.
    #   Select printer, set options, and fire print command.
    # Arguments:
    #   w - widget with contents to print.
    #
    proc ::tk::print::_print {w} {
	variable optlist
	variable option
	variable pinfo
	variable statemap

	# default values for dialog widgets
	option add *Printdialog*TLabel.anchor e
	option add *Printdialog*TMenubutton.Menu.tearOff 0
	option add *Printdialog*TMenubutton.width 12
	option add *Printdialog*TSpinbox.width 12
	# this is tempting to add, but it's better to leave it to
	# user's taste.
	# option add *Printdialog*Menu.background snow

	set class [winfo class $w]
	if {$class ni {Text Canvas}} {
	    return -code error "printing windows of class \"$class\"\
		is not supported"
	}
	# Should this be called with every invocaton?
	# Yes. It allows dynamic discovery of newly added printers
	# whithout having to restart the app
	_setprintenv

	set p ._print
	destroy $p

	# Copy the current values to a dialog's temporary variable.
	# This allow us to cancel the dialog discarding any changes
	# made to the options
	namespace eval dlg {variable option}
	array set dlg::option [array get option]
	set var [namespace which -variable dlg::option]

	# The toplevel of our dialog
	toplevel $p -class Printdialog
	place [ttk::frame $p.background] -x 0 -y 0 -relwidth 1.0 -relheight 1.0
	wm title $p [mc "Print"]
	wm resizable $p 0 0
	wm attributes $p -type dialog
	wm transient $p [winfo toplevel $w]

	# The printer to use
	set pf [ttk::frame $p.printerf]
	pack $pf -side top -fill x -expand no -padx 9p -pady 9p

	ttk::label $pf.printerl -text "[mc "Printer"]"
	set tv [ttk::treeview $pf.prlist -height 5 \
	    -columns {printer location state} \
	    -show headings \
	    -selectmode browse]
	$tv configure \
	    -yscrollcommand [namespace code [list _scroll $pf.sy]] \
	    -xscrollcommand [namespace code [list _scroll $pf.sx]]
	ttk::scrollbar $pf.sy -command [list $tv yview]
	ttk::scrollbar $pf.sx -command [list $tv xview] -orient horizontal
	$tv heading printer  -text [mc "Printer"]
	$tv heading location -text [mc "Location"]
	$tv heading state    -text [mc "State"]
	$tv column printer  -width 200 -stretch 0
	$tv column location -width 100 -stretch 0
	$tv column state    -width 250 -stretch 0

	foreach printer $optlist(printer) {
	    set location [dict getdef $pinfo($printer) printer-location ""]
	    set nstate [dict getdef $pinfo($printer) printer-state 0]
	    set state [dict getdef $statemap $nstate ""]
	    switch -- $nstate {
		3 - 4 {
		    set accepting [dict getdef $pinfo($printer) \
			printer-is-accepting-jobs ""]
		    if {$accepting ne ""} {
			append state ". " [mc "Printer is accepting jobs"]
		    }
		}
		5 {
		    set reason [dict getdef $pinfo($printer) \
			printer-state-reasons ""]
		    if {$reason ne ""} {
			    append state ". (" $reason ")"
		    }
		}
	    }
	    set id [$tv insert {} end \
		-values [list $printer $location $state]]
	    if {$option(printer) eq $printer} {
		$tv selection set $id
	    }
	}

	grid $pf.printerl -sticky w
	grid $pf.prlist $pf.sy -sticky news
	grid $pf.sx -sticky ew
	grid remove $pf.sy $pf.sx
	bind $tv <<TreeviewSelect>> [namespace code {_onselect %W}]

	# Start of printing options
	set of [ttk::labelframe $p.optionsframe -text [mc "Options"]]
	pack $of -fill x -padx 9p -pady {0 9p} -ipadx 2p -ipady 2p

	# COPIES
	ttk::label $of.copiesl -text "[mc "Copies"] :"
	ttk::spinbox $of.copies -textvariable ${var}(copies) \
	    -from 1 -to 1000
	grid $of.copiesl $of.copies -sticky ew -padx 2p -pady 2p
	$of.copies state readonly

	# PAPER SIZE
	ttk::label $of.medial -text "[mc "Paper"] :"
	ttk_optionMenu $of.media ${var}(media) {*}$optlist(media)
	grid $of.medial $of.media -sticky ew -padx 2p -pady 2p

	if {$class eq "Canvas"} {
	    # additional options for Canvas output
	    # SCALE
	    ttk::label $of.percentl -text "[mc "Scale"] :"
	    ttk::spinbox $of.percent -textvariable ${var}(czoom) \
		-from 5 -to 500 -increment 5
	    grid $of.percentl $of.percent -sticky ew -padx 2p -pady 2p
	    $of.percent state readonly

	    # ORIENT
	    ttk::label $of.orientl -text "[mc "Orientation"] :"
	    ttk_optionMenu $of.orient ${var}(orient) {*}$optlist(orient)
	    grid $of.orientl $of.orient -sticky ew -padx 2p -pady 2p

	    # COLOR
	    ttk::label $of.colorl -text "[mc "Output"] :"
	    ttk_optionMenu $of.color ${var}(color) {*}$optlist(color)
	    grid $of.colorl $of.color -sticky ew -padx 2p -pady 2p
	} elseif {$class eq "Text"} {
	    # additional options for Text output
	    # NUMBER-UP
	    ttk::label $of.nupl -text "[mc "Pages per sheet"] :"
	    ttk_optionMenu $of.nup ${var}(number-up) {*}$optlist(number-up)
	    grid $of.nupl $of.nup -sticky ew -padx 2p -pady 2p

	    # TEXT SCALE
	    ttk::label $of.tzooml -text "[mc "Text scale"] :"
	    ttk::spinbox $of.tzoom -textvariable ${var}(tzoom) \
		-from 50 -to 200 -increment 5
	    grid $of.tzooml $of.tzoom -sticky ew -padx 2p -pady 2p
	    $of.tzoom state readonly

	    # PRETTY PRINT (banner on top)
	    ttk::checkbutton $of.pprint -onvalue 1 -offvalue 0 \
		-text [mc "Pretty print"] \
		-variable ${var}(pprint)
	    grid $of.pprint - -sticky ew -padx 2p -pady 2p
	}

	# The buttons frame.
	set bf [ttk::frame $p.buttonf]
	pack $bf -fill x -expand no -side bottom -padx 9p -pady {0 9p}

	ttk::button $bf.print -text [mc "Print"] \
	    -command [namespace code [list _runprint $w $class $p]]
	ttk::button $bf.cancel -text [mc "Cancel"] \
	    -command [list destroy $p]
	pack $bf.print  -side right
	pack $bf.cancel -side right -padx {0 4.5p}

	# cleanup binding
	bind $bf <Destroy> [namespace code [list _cleanup $p]]

	# Center the window as a dialog.
	::tk::PlaceWindow $p
    }

    # _onselect
    #   Updates the selected printer when treeview selection changes.
    # Arguments:
    #   tv - treeview pathname.
    #
    proc ::tk::print::_onselect {tv} {
	variable dlg::option
	set id [$tv selection]
	if {$id eq ""} {
	    # is this even possible?
	    set option(printer) ""
	} else {
	    set option(printer) [$tv set $id printer]
	}
    }

    # _scroll
    #   Implements autoscroll for the printers view
    #
    proc ::tk::print::_scroll {sbar from to} {
	if {$from == 0.0 && $to == 1.0} {
	    grid remove $sbar
	} else {
	    grid $sbar
	    $sbar set $from $to
	}
    }

    # _cleanup
    #   Perform cleanup when the dialog is destroyed.
    # Arguments:
    #   p - print dialog pathname (not used).
    #
    proc ::tk::print::_cleanup {p} {
	namespace delete dlg
    }

    # _runprint -
    #   Execute the print command--print the file.
    # Arguments:
    #   w     - widget with contents to print.
    #   class - class of the widget to print (Canvas or Text).
    #   p     - print dialog pathname.
    #
    proc ::tk::print::_runprint {w class p} {
	variable option
	variable mcmap

	# copy the values back from the dialog
	array set option [array get dlg::option]

	# get (back) name of media from the translated one
	set media [dict get $mcmap(media) $option(media)]
	set printargs {}
	lappend printargs -title "[tk appname]: Tk window $w"
	lappend printargs -copies $option(copies)
	lappend printargs -media $media

	if {$class eq "Canvas"} {
	    set colormode [dict get $mcmap(color) $option(color)]
	    set rotate 0
	    if {[dict get $mcmap(orient) $option(orient)] eq "landscape"} {
		set rotate 1
	    }
	    # Scale based on size of widget, not size of paper.
	    set args {}
	    set sr [$w cget -scrollregion]
	    if {$sr != ""} {
		set sr [lmap x $sr {
		    winfo pixels $w $x
		}]
		foreach k {-x -y -width -height} x $sr {
		    lappend args $k $x
		}
	    }
	    # TODO: is this correct??
	    set printwidth [expr {
		$option(czoom) / 100.0 * [winfo width $w]
	    }]
	    set data [encoding convertto iso8859-1 [$w postscript \
		-colormode $colormode -rotate $rotate -pagewidth $printwidth \
		{*}$args]]
	} elseif {$class eq "Text"} {
	    set tzoom [expr {$option(tzoom) / 100.0}]
	    if {$option(tzoom) != 100} {
		lappend printargs -tzoom $tzoom
	    }
	    if {$option(pprint)} {
		lappend printargs -prettyprint
	    }
	    if {$option(number-up) != 1} {
		lappend printargs -nup $option(number-up)
	    }
	    # these are hardcoded. Should we allow the user to control
	    # margins?
	    lappend printargs -margins [list \
		$option(margin-top)    $option(margin-left) \
		$option(margin-bottom) $option(margin-right) ]
	    # get the data in shape. Cupsfilter's text filter wraps lines
	    # at character level, not words, so we do it by ourselves.
	    # compute usable page width in inches
	    set pw [dict get {a4 8.27 legal 8.5 letter 8.5} $media]
	    set pw [expr {
		$pw - ($option(margin-left) + $option(margin-right)) / 72.0
	    }]
	    # set the wrap length at 98% of computed page width in chars
	    # the 9.8 constant is the product 10.0 (default cpi) * 0.95
	    set wl [expr {int( 9.8 * $pw / $tzoom )}]
	    set data [encoding convertto utf-8 \
		[join [_wrapLines [$w get 1.0 end] $wl] "\n"]]
	}

	# launch the job in the background
	after idle [namespace code \
	    [list cups print $option(printer) $data {*}$printargs]]
	destroy $p
    }
}
#end X11 procedures

namespace eval ::tk::print {
    #begin macOS Aqua procedures
    if {[tk windowingsystem] eq "aqua"} {
	# makePDF -
	#   Convert a file to PDF
	# Arguments:
	#   inFilename -  file containing the data to convert; format is
	#                 autodetected.
	#   outFilename - base for filename to write to; conventionally should
	#                 have .pdf as suffix
	# Returns:
	#   The full pathname of the generated PDF.
	#
	proc makePDF {inFilename outFilename} {
	    set out [::tk::print::makeTempFile $outFilename]
	    try {
		exec /usr/sbin/cupsfilter $inFilename > $out
	    } trap NONE {msg} {
		# cupsfilter produces a lot of debugging output, which we
		# don't want.
		regsub -all -line {^(?:DEBUG|INFO):.*$} $msg "" msg
		set msg [string trimleft [regsub -all {\n+} $msg "\n"] "\n"]
		if {$msg ne ""} {
		    # Lines should be prefixed with WARN or ERROR now
		    puts $msg
		}
	    }
	    return $out
	}
    }
    #end macOS Aqua procedures

    namespace export canvas text
    namespace ensemble create
}

# tk print --
# This procedure prints the canvas and text widgets using platform-
# native API's.
#   Arguments:
#      w: Widget to print.
proc ::tk::print {w} {
    switch [winfo class $w],[tk windowingsystem] {
	"Canvas,win32" -
	"Text,win32" {
	    tailcall ::tk::print::_print $w
	}
	"Canvas,x11" -
	"Text,x11" {
	    tailcall ::tk::print::_print $w
	}
	"Canvas,aqua" {
	    ::tk::print::_printcanvas $w
	    set printfile /tmp/tk_canvas.pdf
	    ::tk::print::_print $printfile
	}
	"Text,aqua" {
	    set txtfile [::tk::print::makeTempFile tk_text.txt [$w get 1.0 end]]
	    try {
		set printfile [::tk::print::makePDF $txtfile [file join /tmp tk_text.pdf]]
		::tk::print::_print $printfile
	    } finally {
		file delete $txtfile
	    }
	}

	default {
	    return -code error -errorcode {TK PRINT CLASS_UNSUPPORTED} \
		"widgets of class [winfo class $w] are not supported on\
		this platform"
	}
    }
}

#Add this command to the tk command ensemble: tk print
#Thanks to Christian Gollwitzer for the guidance here
namespace ensemble configure tk -map \
    [dict merge [namespace ensemble configure tk -map] \
	 {print ::tk::print}]

return

# Local Variables:
# mode: tcl
# fill-column: 78
# End:
