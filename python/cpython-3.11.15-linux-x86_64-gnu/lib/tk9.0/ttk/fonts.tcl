#
# Font specifications.
#
# This file, [source]d at initialization time, sets up the following
# symbolic fonts based on the current platform:
#
# TkDefaultFont	-- default for GUI items not otherwise specified
# TkTextFont	-- font for user text (entry, listbox, others)
# TkFixedFont	-- standard fixed width font
# TkHeadingFont	-- headings (column headings, etc)
# TkCaptionFont -- dialog captions (primary text in alert dialogs, etc.)
# TkTooltipFont	-- font to use for tooltip windows
# TkIconFont	-- font to use for icon captions
# TkMenuFont	-- used to use for menu items
#
# In Tk 8.5, some of these fonts may be provided by the TIP#145 implementation
# (On Windows and Mac OS X as of Oct 2007).
#
# +++ Platform notes:
#
# Windows:
#	The default system font changed from "MS Sans Serif" to "Tahoma"
#	in Windows XP/Windows 2000.
#
#	MS documentation says to use "Tahoma 8" in Windows 2000/XP,
#	although many MS programs still use "MS Sans Serif 8"
#
#	Should use SystemParametersInfo() instead.
#
# Mac OSX / Aqua:
#	Quoth the Apple HIG:
#	The _system font_ (Lucida Grande Regular 13 pt) is used for text
#	in menus, dialogs, and full-size controls.
#	[...] Use the _view font_ (Lucida Grande Regular 12pt) as the default
#	font of text in lists and tables.
#	[...] Use the _emphasized system font_ (Lucida Grande Bold 13 pt)
#	sparingly. It is used for the message text in alerts.
#	[...] The _small system font_ (Lucida Grande Regular 11 pt) [...]
#	is also the default font for column headings in lists, for help tags,
#	and for small controls.
#
#	Note that the font for column headings (TkHeadingFont) is
#	_smaller_ than the default font.
#
#	There does not appear to be any recommendations for fixed-width fonts.
#
# X11:
#	"Classic" look used Helvetica bold for everything except
#	for entry widgets, which use Helvetica medium.
#	Most other toolkits use medium weight for all UI elements,
#	which is what we do now.
#

namespace eval ttk {

variable tip145 [catch {font create TkDefaultFont}]
catch {font create TkTextFont}
catch {font create TkHeadingFont}
catch {font create TkCaptionFont}
catch {font create TkTooltipFont}
catch {font create TkFixedFont}
catch {font create TkIconFont}
catch {font create TkMenuFont}
catch {font create TkSmallCaptionFont}

if {!$tip145} {apply {{} {
global tcl_platform
switch -- [tk windowingsystem] {
    win32 {
	# In safe interps there is no osVersion element.
	if {[info exists tcl_platform(osVersion)]} {
	    if {$tcl_platform(osVersion) >= 5.0} {
		set family "Tahoma"
	    } else {
		set family "MS Sans Serif"
	    }
	} else {
	    if {[lsearch -exact [font families] Tahoma] >= 0} {
		set family "Tahoma"
	    } else {
		set family "MS Sans Serif"
	    }
	}
	set size 8

	font configure TkDefaultFont -family $family -size $size
	font configure TkTextFont    -family $family -size $size
	font configure TkHeadingFont -family $family -size $size
	font configure TkCaptionFont -family $family -size $size -weight bold
	font configure TkTooltipFont -family $family -size $size
	font configure TkFixedFont   -family Courier -size 10
	font configure TkIconFont    -family $family -size $size
	font configure TkMenuFont    -family $family -size $size
	font configure TkSmallCaptionFont -family $family -size $size
    }
    aqua {
	set family "Lucida Grande"
	set fixed "Monaco"
	set menusize 14
	set size 13
	set viewsize 12
	set smallsize 11
	set labelsize 10
	set fixedsize 11

	font configure TkDefaultFont -family $family -size $size
	font configure TkTextFont    -family $family -size $size
	font configure TkHeadingFont -family $family -size $smallsize
	font configure TkCaptionFont -family $family -size $size -weight bold
	font configure TkTooltipFont -family $family -size $smallsize
	font configure TkFixedFont   -family $fixed  -size $fixedsize
	font configure TkIconFont    -family $family -size $size
	font configure TkMenuFont    -family $family -size $menusize
	font configure TkSmallCaptionFont -family $family -size $labelsize
    }
    default -
    x11 {
	if {![catch {tk::pkgconfig get fontsystem} fs] && $fs eq "xft"} {
	    set family "sans-serif"
	    set fixed  "monospace"
	} else {
	    set family "Helvetica"
	    set fixed  "courier"
	}
	if {[::tk::FontScalingFactor] == 1} {
	    set size 10
	    set ttsize 9
	    set capsize 12
	    set fixedsize 10
	} else {
	    set size 20
	    set ttsize 18
	    set capsize 24
	    set fixedsize 20
	}

	font configure TkDefaultFont -family $family -size $size
	font configure TkTextFont    -family $family -size $size
	font configure TkHeadingFont -family $family -size $size    -weight bold
	font configure TkCaptionFont -family $family -size $capsize -weight bold
	font configure TkTooltipFont -family $family -size $ttsize
	font configure TkFixedFont   -family $fixed  -size $fixedsize
	font configure TkIconFont    -family $family -size $size
	font configure TkMenuFont    -family $family -size $size
	font configure TkSmallCaptionFont -family $family -size $ttsize
    }
}
} ::ttk}}

}

#*EOF*
