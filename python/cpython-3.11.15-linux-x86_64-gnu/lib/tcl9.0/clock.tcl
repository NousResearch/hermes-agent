#----------------------------------------------------------------------
#
# clock.tcl --
#
#	This file implements the portions of the [clock] ensemble that are
#	coded in Tcl.  Refer to the users' manual to see the description of
#	the [clock] command and its subcommands.
#
#
#----------------------------------------------------------------------
#
# Copyright © 2004-2007 Kevin B. Kenny
# Copyright © 2015 Sergey G. Brester aka sebres.
# See the file "license.terms" for information on usage and redistribution
# of this file, and for a DISCLAIMER OF ALL WARRANTIES.
#
#----------------------------------------------------------------------

# msgcat 1.7 features are used.

package require msgcat 1.7

# Put the library directory into the namespace for the ensemble so that the
# library code can find message catalogs and time zone definition files.

namespace eval ::tcl::clock \
    [list variable LibDir [info library]]

#----------------------------------------------------------------------
#
# clock --
#
#	Manipulate times.
#
# The 'clock' command manipulates time.  Refer to the user documentation for
# the available subcommands and what they do.
#
#----------------------------------------------------------------------

namespace eval ::tcl::clock {

    # Export the subcommands

    namespace export format
    namespace export clicks
    namespace export microseconds
    namespace export milliseconds
    namespace export scan
    namespace export seconds
    namespace export add

    # Import the message catalog commands that we use.

    namespace import ::msgcat::mclocale
    namespace import ::msgcat::mcpackagelocale

}

#----------------------------------------------------------------------
#
# ::tcl::clock::Initialize --
#
#	Finish initializing the 'clock' subsystem
#
# Results:
#	None.
#
# Side effects:
#	Namespace variable in the 'clock' subsystem are initialized.
#
# The '::tcl::clock::Initialize' procedure initializes the namespace variables
# and root locale message catalog for the 'clock' subsystem.  It is broken
# into a procedure rather than simply evaluated as a script so that it will be
# able to use local variables, avoiding the dangers of 'creative writing' as
# in Bug 1185933.
#
#----------------------------------------------------------------------

proc ::tcl::clock::Initialize {} {

    rename ::tcl::clock::Initialize {}

    variable LibDir

    # Define the Greenwich time zone

    proc InitTZData {} {
	variable TZData
	array unset TZData
	set TZData(:Etc/GMT) {
	    {-9223372036854775808 0 0 GMT}
	}
	set TZData(:GMT) $TZData(:Etc/GMT)
	set TZData(:Etc/UTC) {
	    {-9223372036854775808 0 0 UTC}
	}
	set TZData(:UTC) $TZData(:Etc/UTC)
	set TZData(:localtime) {}
    }
    InitTZData

    mcpackagelocale set {}
    ::msgcat::mcpackageconfig set mcfolder [file join $LibDir msgs]
    ::msgcat::mcpackageconfig set unknowncmd ""
    ::msgcat::mcpackageconfig set changecmd ChangeCurrentLocale

    # Define the message catalog for the root locale.

    ::msgcat::mcmset {} {
	AM {am}
	BCE {B.C.E.}
	CE {C.E.}
	DATE_FORMAT {%m/%d/%Y}
	DATE_TIME_FORMAT {%a %b %e %H:%M:%S %Y}
	DAYS_OF_WEEK_ABBREV	{
	    Sun Mon Tue Wed Thu Fri Sat
	}
	DAYS_OF_WEEK_FULL	{
	    Sunday Monday Tuesday Wednesday Thursday Friday Saturday
	}
	GREGORIAN_CHANGE_DATE	2299161
	LOCALE_DATE_FORMAT {%m/%d/%Y}
	LOCALE_DATE_TIME_FORMAT {%a %b %e %H:%M:%S %Y}
	LOCALE_ERAS {}
	LOCALE_NUMERALS		{
	    00 01 02 03 04 05 06 07 08 09
	    10 11 12 13 14 15 16 17 18 19
	    20 21 22 23 24 25 26 27 28 29
	    30 31 32 33 34 35 36 37 38 39
	    40 41 42 43 44 45 46 47 48 49
	    50 51 52 53 54 55 56 57 58 59
	    60 61 62 63 64 65 66 67 68 69
	    70 71 72 73 74 75 76 77 78 79
	    80 81 82 83 84 85 86 87 88 89
	    90 91 92 93 94 95 96 97 98 99
	}
	LOCALE_TIME_FORMAT {%H:%M:%S}
	LOCALE_YEAR_FORMAT {%EC%Ey}
	MONTHS_ABBREV		{
	    Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec
	}
	MONTHS_FULL		{
		January		February	March
		April		May		June
		July		August		September
		October		November	December
	}
	PM {pm}
	TIME_FORMAT {%H:%M:%S}
	TIME_FORMAT_12 {%I:%M:%S %P}
	TIME_FORMAT_24 {%H:%M}
	TIME_FORMAT_24_SECS {%H:%M:%S}
    }

    # Define a few Gregorian change dates for other locales.  In most cases
    # the change date follows a language, because a nation's colonies changed
    # at the same time as the nation itself.  In many cases, different
    # national boundaries existed; the dominating rule is to follow the
    # nation's capital.

    # Italy, Spain, Portugal, Poland

    ::msgcat::mcset it GREGORIAN_CHANGE_DATE 2299161
    ::msgcat::mcset es GREGORIAN_CHANGE_DATE 2299161
    ::msgcat::mcset pt GREGORIAN_CHANGE_DATE 2299161
    ::msgcat::mcset pl GREGORIAN_CHANGE_DATE 2299161

    # France, Austria

    ::msgcat::mcset fr GREGORIAN_CHANGE_DATE 2299227

    # For Belgium, we follow Southern Netherlands; Liege Diocese changed
    # several weeks later.

    ::msgcat::mcset fr_BE GREGORIAN_CHANGE_DATE 2299238
    ::msgcat::mcset nl_BE GREGORIAN_CHANGE_DATE 2299238

    # Austria

    ::msgcat::mcset de_AT GREGORIAN_CHANGE_DATE 2299527

    # Hungary

    ::msgcat::mcset hu GREGORIAN_CHANGE_DATE 2301004

    # Germany, Norway, Denmark (Catholic Germany changed earlier)

    ::msgcat::mcset de_DE GREGORIAN_CHANGE_DATE 2342032
    ::msgcat::mcset nb GREGORIAN_CHANGE_DATE 2342032
    ::msgcat::mcset nn GREGORIAN_CHANGE_DATE 2342032
    ::msgcat::mcset no GREGORIAN_CHANGE_DATE 2342032
    ::msgcat::mcset da GREGORIAN_CHANGE_DATE 2342032

    # Holland (Brabant, Gelderland, Flanders, Friesland, etc. changed at
    # various times)

    ::msgcat::mcset nl GREGORIAN_CHANGE_DATE 2342165

    # Protestant Switzerland (Catholic cantons changed earlier)

    ::msgcat::mcset fr_CH GREGORIAN_CHANGE_DATE 2361342
    ::msgcat::mcset it_CH GREGORIAN_CHANGE_DATE 2361342
    ::msgcat::mcset de_CH GREGORIAN_CHANGE_DATE 2361342

    # English speaking countries

    ::msgcat::mcset en GREGORIAN_CHANGE_DATE 2361222

    # Sweden (had several changes onto and off of the Gregorian calendar)

    ::msgcat::mcset sv GREGORIAN_CHANGE_DATE 2361390

    # Russia

    ::msgcat::mcset ru GREGORIAN_CHANGE_DATE 2421639

    # Romania (Transylvania changed earlier - perhaps de_RO should show the
    # earlier date?)

    ::msgcat::mcset ro GREGORIAN_CHANGE_DATE 2422063

    # Greece

    ::msgcat::mcset el GREGORIAN_CHANGE_DATE 2423480

    #------------------------------------------------------------------
    #
    #				CONSTANTS
    #
    #------------------------------------------------------------------

    # Paths at which binary time zone data for the Olson libraries are known
    # to reside on various operating systems

    variable ZoneinfoPaths {}
    foreach path {
	/usr/share/zoneinfo
	/usr/share/lib/zoneinfo
	/usr/lib/zoneinfo
	/usr/local/etc/zoneinfo
    } {
	if { [file isdirectory $path] } {
	    lappend ZoneinfoPaths $path
	}
    }

    # Define the directories for time zone data and message catalogs.

    variable DataDir [file join $LibDir tzdata]

    # Number of days in the months, in common years and leap years.

    variable DaysInRomanMonthInCommonYear \
	{ 31 28 31 30 31 30 31 31 30 31 30 31 }
    variable DaysInRomanMonthInLeapYear \
	{ 31 29 31 30 31 30 31 31 30 31 30 31 }
    variable DaysInPriorMonthsInCommonYear [list 0]
    variable DaysInPriorMonthsInLeapYear [list 0]
    set i 0
    foreach j $DaysInRomanMonthInCommonYear {
	lappend DaysInPriorMonthsInCommonYear [incr i $j]
    }
    set i 0
    foreach j $DaysInRomanMonthInLeapYear {
	lappend DaysInPriorMonthsInLeapYear [incr i $j]
    }

    # Another epoch (Hi, Jeff!)

    variable Roddenberry 1946

    # Integer ranges

    variable MINWIDE -9223372036854775808
    variable MAXWIDE 9223372036854775807

    # Day before Leap Day

    variable FEB_28	       58

    # Default configuration

    ::tcl::unsupported::clock::configure -current-locale  [mclocale]
    #::tcl::unsupported::clock::configure -default-locale  C
    #::tcl::unsupported::clock::configure -year-century    2000 \
    #          -century-switch  38

    # Translation table to map Windows TZI onto cities, so that the Olson
    # rules can apply.  In some cases the mapping is ambiguous, so it's wise
    # to specify $::env(TCL_TZ) rather than simply depending on the system
    # time zone.

    # The keys are long lists of values obtained from the time zone
    # information in the Registry.  In order, the list elements are:
    #   Bias StandardBias DaylightBias
    #   StandardDate.wYear StandardDate.wMonth StandardDate.wDayOfWeek
    #   StandardDate.wDay StandardDate.wHour StandardDate.wMinute
    #   StandardDate.wSecond StandardDate.wMilliseconds
    #   DaylightDate.wYear DaylightDate.wMonth DaylightDate.wDayOfWeek
    #   DaylightDate.wDay DaylightDate.wHour DaylightDate.wMinute
    #   DaylightDate.wSecond DaylightDate.wMilliseconds
    # The values are the names of time zones where those rules apply.  There
    # is considerable ambiguity in certain zones; an attempt has been made to
    # make a reasonable guess, but this table needs to be taken with a grain
    # of salt.

    variable WinZoneInfo [dict create {*}{
	{-43200 0 3600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0}  :Pacific/Kwajalein
	{-39600 0 3600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0}	 :Pacific/Midway
	{-36000 0 3600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0}  :Pacific/Honolulu
	{-32400 0 3600 0 11 0 1 2 0 0 0 0 3 0 2 2 0 0 0} :America/Anchorage
	{-28800 0 3600 0 11 0 1 2 0 0 0 0 3 0 2 2 0 0 0} :America/Los_Angeles
	{-28800 0 3600 0 10 0 5 2 0 0 0 0 4 0 1 2 0 0 0} :America/Tijuana
	{-25200 0 3600 0 11 0 1 2 0 0 0 0 3 0 2 2 0 0 0} :America/Denver
	{-25200 0 3600 0 10 0 5 2 0 0 0 0 4 0 1 2 0 0 0} :America/Chihuahua
	{-25200 0 3600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0}  :America/Phoenix
	{-21600 0 3600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0}  :America/Regina
	{-21600 0 3600 0 11 0 1 2 0 0 0 0 3 0 2 2 0 0 0} :America/Chicago
	{-21600 0 3600 0 10 0 5 2 0 0 0 0 4 0 1 2 0 0 0} :America/Mexico_City
	{-18000 0 3600 0 11 0 1 2 0 0 0 0 3 0 2 2 0 0 0} :America/New_York
	{-18000 0 3600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0}  :America/Indianapolis
	{-14400 0 3600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0}  :America/Caracas
	{-14400 0 3600 0 3 6 2 23 59 59 999 0 10 6 2 23 59 59 999}
							 :America/Santiago
	{-14400 0 3600 0 2 0 5 2 0 0 0 0 11 0 1 2 0 0 0} :America/Manaus
	{-14400 0 3600 0 11 0 1 2 0 0 0 0 3 0 2 2 0 0 0} :America/Halifax
	{-12600 0 3600 0 10 0 5 2 0 0 0 0 4 0 1 2 0 0 0} :America/St_Johns
	{-10800 0 3600 0 2 0 2 2 0 0 0 0 10 0 3 2 0 0 0} :America/Sao_Paulo
	{-10800 0 3600 0 10 0 5 2 0 0 0 0 4 0 1 2 0 0 0} :America/Godthab
	{-10800 0 3600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0}  :America/Buenos_Aires
	{-10800 0 3600 0 2 0 5 2 0 0 0 0 11 0 1 2 0 0 0} :America/Bahia
	{-10800 0 3600 0 3 0 2 2 0 0 0 0 10 0 1 2 0 0 0} :America/Montevideo
	{-7200 0 3600 0 9 0 5 2 0 0 0 0 3 0 5 2 0 0 0}   :America/Noronha
	{-3600 0 3600 0 10 0 5 3 0 0 0 0 3 0 5 2 0 0 0}  :Atlantic/Azores
	{-3600 0 3600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0}   :Atlantic/Cape_Verde
	{0 0 3600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0}       :UTC
	{0 0 3600 0 10 0 5 2 0 0 0 0 3 0 5 1 0 0 0}      :Europe/London
	{3600 0 3600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0}    :Africa/Kinshasa
	{3600 0 3600 0 10 0 5 3 0 0 0 0 3 0 5 2 0 0 0}   :CET
	{7200 0 3600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0}    :Africa/Harare
	{7200 0 3600 0 9 4 5 23 59 59 0 0 4 4 5 23 59 59 0}
							 :Africa/Cairo
	{7200 0 3600 0 10 0 5 4 0 0 0 0 3 0 5 3 0 0 0}   :Europe/Helsinki
	{7200 0 3600 0 9 0 3 2 0 0 0 0 3 5 5 2 0 0 0}    :Asia/Jerusalem
	{7200 0 3600 0 9 0 5 1 0 0 0 0 3 0 5 0 0 0 0}    :Europe/Bucharest
	{7200 0 3600 0 10 0 5 3 0 0 0 0 3 0 5 2 0 0 0}   :Europe/Athens
	{7200 0 3600 0 9 5 5 1 0 0 0 0 3 4 5 0 0 0 0}    :Asia/Amman
	{7200 0 3600 0 10 6 5 23 59 59 999 0 3 0 5 0 0 0 0}
							 :Asia/Beirut
	{7200 0 -3600 0 4 0 1 2 0 0 0 0 9 0 1 2 0 0 0}   :Africa/Windhoek
	{10800 0 3600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0}   :Asia/Riyadh
	{10800 0 3600 0 10 0 1 4 0 0 0 0 4 0 1 3 0 0 0}  :Asia/Baghdad
	{10800 0 3600 0 10 0 5 3 0 0 0 0 3 0 5 2 0 0 0}  :Europe/Moscow
	{12600 0 3600 0 9 2 4 2 0 0 0 0 3 0 1 2 0 0 0}   :Asia/Tehran
	{14400 0 3600 0 10 0 5 5 0 0 0 0 3 0 5 4 0 0 0}  :Asia/Baku
	{14400 0 3600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0}   :Asia/Muscat
	{14400 0 3600 0 10 0 5 3 0 0 0 0 3 0 5 2 0 0 0}  :Asia/Tbilisi
	{16200 0 3600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0}   :Asia/Kabul
	{18000 0 3600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0}   :Asia/Karachi
	{18000 0 3600 0 10 0 5 3 0 0 0 0 3 0 5 2 0 0 0}  :Asia/Yekaterinburg
	{19800 0 3600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0}   :Asia/Calcutta
	{20700 0 3600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0}   :Asia/Katmandu
	{21600 0 3600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0}   :Asia/Dhaka
	{21600 0 3600 0 10 0 5 3 0 0 0 0 3 0 5 2 0 0 0}  :Asia/Novosibirsk
	{23400 0 3600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0}   :Asia/Rangoon
	{25200 0 3600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0}   :Asia/Bangkok
	{25200 0 3600 0 10 0 5 3 0 0 0 0 3 0 5 2 0 0 0}  :Asia/Krasnoyarsk
	{28800 0 3600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0}   :Asia/Chongqing
	{28800 0 3600 0 10 0 5 3 0 0 0 0 3 0 5 2 0 0 0}  :Asia/Irkutsk
	{32400 0 3600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0}   :Asia/Tokyo
	{32400 0 3600 0 10 0 5 3 0 0 0 0 3 0 5 2 0 0 0}  :Asia/Yakutsk
	{34200 0 3600 0 3 0 5 3 0 0 0 0 10 0 5 2 0 0 0}  :Australia/Adelaide
	{34200 0 3600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0}   :Australia/Darwin
	{36000 0 3600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0}   :Australia/Brisbane
	{36000 0 3600 0 10 0 5 3 0 0 0 0 3 0 5 2 0 0 0}  :Asia/Vladivostok
	{36000 0 3600 0 3 0 5 3 0 0 0 0 10 0 1 2 0 0 0}  :Australia/Hobart
	{36000 0 3600 0 3 0 5 3 0 0 0 0 10 0 5 2 0 0 0}  :Australia/Sydney
	{39600 0 3600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0}   :Pacific/Noumea
	{43200 0 3600 0 3 0 3 3 0 0 0 0 10 0 1 2 0 0 0}  :Pacific/Auckland
	{43200 0 3600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0}   :Pacific/Fiji
	{46800 0 3600 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0}   :Pacific/Tongatapu
    }]

    # Legacy time zones, used primarily for parsing RFC822 dates.

    variable LegacyTimeZone [dict create \
	gmt	+0000 \
	ut	+0000 \
	utc	+0000 \
	bst	+0100 \
	wet	+0000 \
	wat	-0100 \
	at	-0200 \
	nft	-0330 \
	nst	-0330 \
	ndt	-0230 \
	ast	-0400 \
	adt	-0300 \
	est	-0500 \
	edt	-0400 \
	cst	-0600 \
	cdt	-0500 \
	mst	-0700 \
	mdt	-0600 \
	pst	-0800 \
	pdt	-0700 \
	yst	-0900 \
	ydt	-0800 \
	akst	-0900 \
	akdt	-0800 \
	hst	-1000 \
	hdt	-0900 \
	cat	-1000 \
	ahst	-1000 \
	nt	-1100 \
	idlw	-1200 \
	cet	+0100 \
	cest	+0200 \
	met	+0100 \
	mewt	+0100 \
	mest	+0200 \
	swt	+0100 \
	sst	+0200 \
	fwt	+0100 \
	fst	+0200 \
	eet	+0200 \
	eest	+0300 \
	bt	+0300 \
	it	+0330 \
	zp4	+0400 \
	zp5	+0500 \
	ist	+0530 \
	zp6	+0600 \
	wast	+0700 \
	wadt	+0800 \
	jt	+0730 \
	cct	+0800 \
	jst	+0900 \
	kst     +0900 \
	cast	+0930 \
	jdt     +1000 \
	kdt     +1000 \
	cadt	+1030 \
	east	+1000 \
	eadt	+1030 \
	gst	+1000 \
	nzt	+1200 \
	nzst	+1200 \
	nzdt	+1300 \
	idle	+1200 \
	a	+0100 \
	b	+0200 \
	c	+0300 \
	d	+0400 \
	e	+0500 \
	f	+0600 \
	g	+0700 \
	h	+0800 \
	i	+0900 \
	k	+1000 \
	l	+1100 \
	m	+1200 \
	n	-0100 \
	o	-0200 \
	p	-0300 \
	q	-0400 \
	r	-0500 \
	s	-0600 \
	t	-0700 \
	u	-0800 \
	v	-0900 \
	w	-1000 \
	x	-1100 \
	y	-1200 \
	z	+0000 \
    ]

    # Caches

    variable LocFmtMap [dict create];	# Dictionary with localized format maps

    variable TimeZoneBad [dict create]; # Dictionary whose keys are time zone
					# names and whose values are 1 if
					# the time zone is unknown and 0
					# if it is known.
    variable TZData;			# Array whose keys are time zone names
					# and whose values are lists of quads
					# comprising start time, UTC offset,
					# Daylight Saving Time indicator, and
					# time zone abbreviation.

    variable mcLocales	 [dict create];	# Dictionary with loaded locales
    variable mcMergedCat [dict create];	# Dictionary with merged locale catalogs
}
::tcl::clock::Initialize

#----------------------------------------------------------------------

# mcget --
#
#	Return the merged translation catalog for the ::tcl::clock namespace
#	Searching of catalog is similar to "msgcat::mc".
#
#	Contrary to "msgcat::mc" may additionally load a package catalog
#	on demand.
#
# Arguments:
#	loc	The locale used for translation.
#
# Results:
#	Returns the dictionary object as whole catalog of the package/locale.
#
proc ::tcl::clock::mcget {loc} {
    variable mcMergedCat
    switch -- $loc system {
	set loc [GetSystemLocale]
    } current {
	set loc [mclocale]
    }
    if {$loc ne {}} {
	set loc [string tolower $loc]
    }

    # try to retrieve now if already available:
    if {[dict exists $mcMergedCat $loc]} {
	return [dict get $mcMergedCat $loc]
    }

    # get locales list for given locale (de_de -> {de_de de {}})
    variable mcLocales
    if {[dict exists $mcLocales $loc]} {
	set loclist [dict get $mcLocales $loc]
    } else {
	# save current locale:
	set prevloc [mclocale]
	# lazy load catalog on demand (set it will load the catalog)
	mcpackagelocale set $loc
	set loclist [msgcat::mcutil::getpreferences $loc]
	dict set $mcLocales $loc $loclist
	# restore:
	if {$prevloc ne $loc} {
	   mcpackagelocale set $prevloc
	}
    }
    # get whole catalog:
    mcMerge $loclist
}

# mcMerge --
#
#	Merge message catalog dictionaries to one dictionary.
#
# Arguments:
#	locales		List of locales to merge.
#
# Results:
#	Returns the (weak pointer) to merged dictionary of message catalog.
#
proc ::tcl::clock::mcMerge {locales} {
    variable mcMergedCat
    if {[dict exists $mcMergedCat [set loc [lindex $locales 0]]]} {
	return [dict get $mcMergedCat $loc]
    }
    # package msgcat currently does not provide possibility to get whole catalog:
    upvar ::msgcat::Msgs Msgs
    set ns ::tcl::clock
    # Merge sequential locales (in reverse order, e. g. {} -> en -> en_en):
    if {[llength $locales] > 1} {
	set mrgcat [mcMerge [lrange $locales 1 end]]
	if {[dict exists $Msgs $ns $loc]} {
	    set mrgcat [dict merge $mrgcat [dict get $Msgs $ns $loc]]
	    dict set mrgcat L $loc
	    # remove any previously localized formats (merged from parent
	    # locale and possibly cached in parent-mc by ClockLocalizeFormat),
	    # because they may depend on values which may vary in derivate:
	    foreach k [dict keys $mrgcat] {
		if {[string match FMT_* $k]} { dict unset mrgcat $k }
	    }
	} else {
	    # be sure a duplicate is created, don't overwrite {} (common) locale:
	    set mrgcat [dict merge $mrgcat [dict create L $loc]]
	}
    } else {
	if {[dict exists $Msgs $ns $loc]} {
	    set mrgcat [dict get $Msgs $ns $loc]
	    dict set mrgcat L $loc
	} else {
	    # be sure a duplicate is created, don't overwrite {} (common) locale:
	    set mrgcat [dict create L $loc]
	}
    }
    dict set mcMergedCat $loc $mrgcat
    # return smart reference (shared dict as object with exact one ref-counter)
    return $mrgcat
}

#----------------------------------------------------------------------
#
# GetSystemLocale --
#
#	Determines the system locale, which corresponds to "system"
#	keyword for locale parameter of 'clock' command.
#
# Parameters:
#	None.
#
# Results:
#	Returns the system locale.
#
# Side effects:
#	None.
#
#----------------------------------------------------------------------

proc ::tcl::clock::GetSystemLocale {} {
    if { $::tcl_platform(platform) ne {windows} } {
	# On a non-windows platform, the 'system' locale is the same as
	# the 'current' locale

	return [mclocale]
    }

    # On a windows platform, the 'system' locale is adapted from the
    # 'current' locale by applying the date and time formats from the
    # Control Panel.  First, load the 'current' locale if it's not yet
    # loaded

    mcpackagelocale set [mclocale]

    # Make a new locale string for the system locale, and get the
    # Control Panel information

    set locale [mclocale]_windows
    if { ! [mcpackagelocale present $locale] } {
	LoadWindowsDateTimeFormats $locale
    }

    return $locale
}

#----------------------------------------------------------------------
#
# EnterLocale --
#
#	Switch [mclocale] to a given locale if necessary
#
# Parameters:
#	locale -- Desired locale
#
# Results:
#	Returns the locale that was previously current.
#
# Side effects:
#	Does [mclocale].  If necessary, loades the designated locale's files.
#
#----------------------------------------------------------------------

proc ::tcl::clock::EnterLocale { locale } {
    switch -- $locale system {
	set locale [GetSystemLocale]
    } current {
	set locale [mclocale]
    }
    # Select the locale, eventually load it
    mcpackagelocale set $locale
    return $locale
}

#----------------------------------------------------------------------
#
# _registryExists --
# _hasRegistry --
#
#	Helpers that checks whether registry module is available (Windows only)
#	and loads it on demand.
#
# Side effects:
#	_hasRegistry does it only once, and hereafter simply returns 1 or 0.
#
#----------------------------------------------------------------------
proc ::tcl::clock::_registryExists {} {
    if { $::tcl_platform(platform) eq {windows} } {
	if { [catch { package require registry 1.3 }] } {
	    # try to load registry directly from root (if uninstalled / development env):
	    if {[regexp {[/\\]library$} [info library]]} {catch {
		load [lindex \
			[glob -tails -directory [file dirname [info nameofexecutable]] \
			    tcl9registry*[expr {[::tcl::pkgconfig get debug] ? {g} : {}}].dll] 0 \
		] Registry
	    }}
	}
	if { [namespace which -command ::registry] ne "" } {
	    return 1
	}
    }
    return 0
}
proc ::tcl::clock::_hasRegistry {} {
    set res [_registryExists]
    proc ::tcl::clock::_hasRegistry {} [list return $res]
    return $res
}

#----------------------------------------------------------------------
#
# LoadWindowsDateTimeFormats --
#
#	Load the date/time formats from the Control Panel in Windows and
#	convert them so that they're usable by Tcl.
#
# Parameters:
#	locale - Name of the locale in whose message catalog
#	         the converted formats are to be stored.
#
# Results:
#	None.
#
# Side effects:
#	Updates the given message catalog with the locale strings.
#
# Presumes that on entry, [mclocale] is set to the current locale, so that
# default strings can be obtained if the Registry query fails.
#
#----------------------------------------------------------------------

proc ::tcl::clock::LoadWindowsDateTimeFormats { locale } {
    # Bail out if we can't find the Registry

    if { ![_hasRegistry] } return

    if { ![catch {
	registry get "HKEY_CURRENT_USER\\Control Panel\\International" \
	    sShortDate
    } string] } {
	set quote {}
	set datefmt {}
	foreach { unquoted quoted } [split $string '] {
	    append datefmt $quote [string map {
		dddd %A
		ddd  %a
		dd   %d
		d    %e
		MMMM %B
		MMM  %b
		MM   %m
		M    %N
		yyyy %Y
		yy   %y
		y    %y
		gg   {}
	    } $unquoted]
	    if { $quoted eq {} } {
		set quote '
	    } else {
		set quote $quoted
	    }
	}
	::msgcat::mcset $locale DATE_FORMAT $datefmt
    }

    if { ![catch {
	registry get "HKEY_CURRENT_USER\\Control Panel\\International" \
	    sLongDate
    } string] } {
	set quote {}
	set ldatefmt {}
	foreach { unquoted quoted } [split $string '] {
	    append ldatefmt $quote [string map {
		dddd %A
		ddd  %a
		dd   %d
		d    %e
		MMMM %B
		MMM  %b
		MM   %m
		M    %N
		yyyy %Y
		yy   %y
		y    %y
		gg   {}
	    } $unquoted]
	    if { $quoted eq {} } {
		set quote '
	    } else {
		set quote $quoted
	    }
	}
	::msgcat::mcset $locale LOCALE_DATE_FORMAT $ldatefmt
    }

    if { ![catch {
	registry get "HKEY_CURRENT_USER\\Control Panel\\International" \
	    sTimeFormat
    } string] } {
	set quote {}
	set timefmt {}
	foreach { unquoted quoted } [split $string '] {
	    append timefmt $quote [string map {
		HH    %H
		H     %k
		hh    %I
		h     %l
		mm    %M
		m     %M
		ss    %S
		s     %S
		tt    %p
		t     %p
	    } $unquoted]
	    if { $quoted eq {} } {
		set quote '
	    } else {
		set quote $quoted
	    }
	}
	::msgcat::mcset $locale TIME_FORMAT $timefmt
    }

    catch {
	::msgcat::mcset $locale DATE_TIME_FORMAT "$datefmt $timefmt"
    }
    catch {
	::msgcat::mcset $locale LOCALE_DATE_TIME_FORMAT "$ldatefmt $timefmt"
    }

    return

}

#----------------------------------------------------------------------
#
# LocalizeFormat --
#
#	Map away locale-dependent format groups in a clock format.
#
# Parameters:
#	locale -- Current [mclocale] locale, supplied to avoid
#		  an extra call
#	format -- Format supplied to [clock scan] or [clock format]
#	mcd    -- Message catalog dictionary for current locale (read-only,
#		  don't store it to avoid shared references).
#
# Results:
#	Returns the string with locale-dependent composite format groups
#	substituted out.
#
# Side effects:
#	None.
#
#----------------------------------------------------------------------

proc ::tcl::clock::LocalizeFormat { locale format mcd } {
    variable LocFmtMap

    # get map list cached or build it:
    if {[dict exists $LocFmtMap $locale]} {
	set mlst [dict get $LocFmtMap $locale]
    } else {
	# Handle locale-dependent format groups by mapping them out of the format
	# string.  Note that the order of the [string map] operations is
	# significant because later formats can refer to later ones; for example
	# %c can refer to %X, which in turn can refer to %T.

	set mlst {
	    %% %%
	    %D %m/%d/%Y
	    %+ {%a %b %e %H:%M:%S %Z %Y}
	}
	lappend mlst %EY [string map $mlst [dict get $mcd LOCALE_YEAR_FORMAT]]
	lappend mlst %T  [string map $mlst [dict get $mcd TIME_FORMAT_24_SECS]]
	lappend mlst %R  [string map $mlst [dict get $mcd TIME_FORMAT_24]]
	lappend mlst %r  [string map $mlst [dict get $mcd TIME_FORMAT_12]]
	lappend mlst %X  [string map $mlst [dict get $mcd TIME_FORMAT]]
	lappend mlst %EX [string map $mlst [dict get $mcd LOCALE_TIME_FORMAT]]
	lappend mlst %x  [string map $mlst [dict get $mcd DATE_FORMAT]]
	lappend mlst %Ex [string map $mlst [dict get $mcd LOCALE_DATE_FORMAT]]
	lappend mlst %c  [string map $mlst [dict get $mcd DATE_TIME_FORMAT]]
	lappend mlst %Ec [string map $mlst [dict get $mcd LOCALE_DATE_TIME_FORMAT]]

	dict set LocFmtMap $locale $mlst
    }

    # translate copy of format (don't use format object here, because otherwise
    # it can lose its internal representation (string map - convert to unicode)
    set locfmt [string map $mlst [string range " $format" 1 end]]

    # Save original format as long as possible, because of internal
    # representation (performance).
    # Note that in this case such format will be never localized (also
    # using another locales). To prevent this return a duplicate (but
    # it may be slower).
    if {$locfmt eq $format} {
	set locfmt $format
    }

    return $locfmt
}

#----------------------------------------------------------------------
#
# GetSystemTimeZone --
#
#	Determines the system time zone, which is the default for the
#	'clock' command if no other zone is supplied.
#
# Parameters:
#	None.
#
# Results:
#	Returns the system time zone.
#
# Side effects:
#	Stores the system time zone in engine configuration, since
#	determining it may be an expensive process.
#
#----------------------------------------------------------------------

proc ::tcl::clock::GetSystemTimeZone {} {
    variable TimeZoneBad

    if {[set result [getenv TCL_TZ]] ne {}} {
	set timezone $result
    } elseif {[set result [getenv TZ]] ne {}} {
	set timezone $result
    } else {
	# ask engine for the cached timezone:
	set timezone [::tcl::unsupported::clock::configure -system-tz]
	if { $timezone ne "" } {
	    return $timezone
	}
	if { $::tcl_platform(platform) eq {windows} } {
	    set timezone [GuessWindowsTimeZone]
	} elseif { [file exists /etc/localtime]
		   && ![catch {ReadZoneinfoFile \
				   Tcl/Localtime /etc/localtime}] } {
	    set timezone :Tcl/Localtime
	} else {
	    set timezone :localtime
	}
    }
    if { ![dict exists $TimeZoneBad $timezone] } {
	catch {set timezone [SetupTimeZone $timezone]}
    }

    if { [dict exists $TimeZoneBad $timezone] } {
	set timezone :localtime
    }

    # tell backend - current system timezone:
    ::tcl::unsupported::clock::configure -system-tz $timezone

    return $timezone
}

#----------------------------------------------------------------------
#
# SetupTimeZone --
#
#	Given the name or specification of a time zone, sets up its in-memory
#	data.
#
# Parameters:
#	tzname - Name of a time zone
#
# Results:
#	Unless the time zone is ':localtime', sets the TZData array to contain
#	the lookup table for local<->UTC conversion.  Returns an error if the
#	time zone cannot be parsed.
#
#----------------------------------------------------------------------

proc ::tcl::clock::SetupTimeZone { timezone {alias {}} } {
    variable TZData

    if {! [info exists TZData($timezone)] } {

	variable TimeZoneBad
	if { [dict exists $TimeZoneBad $timezone] } {
	    return -code error \
		-errorcode [list CLOCK badTimeZone $timezone] \
		"time zone \"$timezone\" not found"
	}
	variable MINWIDE
	if {
	    [regexp {^([-+])(\d\d)(?::?(\d\d)(?::?(\d\d))?)?} $timezone \
		    -> s hh mm ss]
	} then {
	    # Make a fixed offset

	    ::scan $hh %d hh
	    if { $mm eq {} } {
		set mm 0
	    } else {
		::scan $mm %d mm
	    }
	    if { $ss eq {} } {
		set ss 0
	    } else {
		::scan $ss %d ss
	    }
	    set offset [expr { ( $hh * 60 + $mm ) * 60 + $ss }]
	    if { $s eq {-} } {
		set offset [expr { - $offset }]
	    }
	    set TZData($timezone) [list [list $MINWIDE $offset -1 $timezone]]

	} elseif { [string index $timezone 0] eq {:} } {
	    # Convert using a time zone file

	    if {
		[catch {
		    LoadTimeZoneFile [string range $timezone 1 end]
		}] && [catch {
		    LoadZoneinfoFile [string range $timezone 1 end]
		} ret opts]
	    } then {
		dict unset opts -errorinfo
		if {[lindex [dict get $opts -errorcode] 0] ne "CLOCK"} {
		    dict set opts -errorcode [list CLOCK badTimeZone $timezone]
		    set ret "time zone \"$timezone\" not found: $ret"
		}
		dict set TimeZoneBad $timezone 1
		return -options $opts $ret
	    }
	} elseif { ![catch {ParsePosixTimeZone $timezone} tzfields] } {
	    # This looks like a POSIX time zone - try to process it

	    if { [catch {ProcessPosixTimeZone $tzfields} ret opts] } {
		dict unset opts -errorinfo
		if {[lindex [dict get $opts -errorcode] 0] ne "CLOCK"} {
		    dict set opts -errorcode [list CLOCK badTimeZone $timezone]
		    set ret "time zone \"$timezone\" not found: $ret"
		}
		dict set TimeZoneBad $timezone 1
		return -options $opts $ret
	    } else {
		set TZData($timezone) $ret
	    }

	} else {

	    variable LegacyTimeZone

	    # We couldn't parse this as a POSIX time zone.  Try again with a
	    # time zone file - this time without a colon

	    if { [catch { LoadTimeZoneFile $timezone }]
		 && [catch { LoadZoneinfoFile $timezone } ret opts] } {

		# Check may be a legacy zone:

		if { $alias eq {} && ![catch {
		    set tzname [dict get $LegacyTimeZone [string tolower $timezone]]
		}] } {
		    set tzname [::tcl::clock::SetupTimeZone $tzname $timezone]
		    set TZData($timezone) $TZData($tzname)
		    # tell backend - timezone is initialized and return shared timezone object:
		    return [::tcl::unsupported::clock::configure -setup-tz $timezone]
		}

		dict unset opts -errorinfo
		if {[lindex [dict get $opts -errorcode] 0] ne "CLOCK"} {
		    dict set opts -errorcode [list CLOCK badTimeZone $timezone]
		    set ret "time zone \"$timezone\" not found: $ret"
		}
		dict set TimeZoneBad $timezone 1
		return -options $opts $ret
	    }
	    set TZData($timezone) $TZData(:$timezone)
	}
    }

    # tell backend - timezone is initialized and return shared timezone object:
    ::tcl::unsupported::clock::configure -setup-tz $timezone
}

#----------------------------------------------------------------------
#
# GuessWindowsTimeZone --
#
#	Determines the system time zone on windows.
#
# Parameters:
#	None.
#
# Results:
#	Returns a time zone specifier that corresponds to the system time zone
#	information found in the Registry.
#
# Bugs:
#	Fixed dates for DST change are unimplemented at present, because no
#	time zone information supplied with Windows actually uses them!
#
# On a Windows system where neither $env(TCL_TZ) nor $env(TZ) is specified,
# GuessWindowsTimeZone looks in the Registry for the system time zone
# information.  It then attempts to find an entry in WinZoneInfo for a time
# zone that uses the same rules.  If it finds one, it returns it; otherwise,
# it constructs a Posix-style time zone string and returns that.
#
#----------------------------------------------------------------------

proc ::tcl::clock::GuessWindowsTimeZone {} {
    variable WinZoneInfo
    variable TimeZoneBad

    if { ![_hasRegistry] } {
	return :localtime
    }

    # Dredge time zone information out of the registry

    if { [catch {
	set rpath HKEY_LOCAL_MACHINE\\System\\CurrentControlSet\\Control\\TimeZoneInformation
	set data [list \
		      [expr { -60
			      * [registry get $rpath Bias] }] \
		      [expr { -60
				  * [registry get $rpath StandardBias] }] \
		      [expr { -60 \
				  * [registry get $rpath DaylightBias] }]]
	set stdtzi [registry get $rpath StandardStart]
	foreach ind {0 2 14 4 6 8 10 12} {
	    binary scan $stdtzi @${ind}s val
	    lappend data $val
	}
	set daytzi [registry get $rpath DaylightStart]
	foreach ind {0 2 14 4 6 8 10 12} {
	    binary scan $daytzi @${ind}s val
	    lappend data $val
	}
    }] } {
	# Missing values in the Registry - bail out

	return :localtime
    }

    # Make up a Posix time zone specifier if we can't find one.  Check here
    # that the tzdata file exists, in case we're running in an environment
    # (e.g. starpack) where tzdata is incomplete.  (Bug 1237907)

    if { [dict exists $WinZoneInfo $data] } {
	set tzname [dict get $WinZoneInfo $data]
	if { ! [dict exists $TimeZoneBad $tzname] } {
	    catch {set tzname [SetupTimeZone $tzname]}
	}
    } else {
	set tzname {}
    }
    if { $tzname eq {} || [dict exists $TimeZoneBad $tzname] } {
	lassign $data \
	    bias stdBias dstBias \
	    stdYear stdMonth stdDayOfWeek stdDayOfMonth \
	    stdHour stdMinute stdSecond stdMillisec \
	    dstYear dstMonth dstDayOfWeek dstDayOfMonth \
	    dstHour dstMinute dstSecond dstMillisec
	set stdDelta [expr { $bias + $stdBias }]
	set dstDelta [expr { $bias + $dstBias }]
	if { $stdDelta <= 0 } {
	    set stdSignum +
	    set stdDelta [expr { - $stdDelta }]
	    set dispStdSignum -
	} else {
	    set stdSignum -
	    set dispStdSignum +
	}
	set hh [::format %02d [expr { $stdDelta / 3600 }]]
	set mm [::format %02d [expr { ($stdDelta / 60 ) % 60 }]]
	set ss [::format %02d [expr { $stdDelta % 60 }]]
	set tzname {}
	append tzname < $dispStdSignum $hh $mm > $stdSignum $hh : $mm : $ss
	if { $stdMonth >= 0 } {
	    if { $dstDelta <= 0 } {
		set dstSignum +
		set dstDelta [expr { - $dstDelta }]
		set dispDstSignum -
	    } else {
		set dstSignum -
		set dispDstSignum +
	    }
	    set hh [::format %02d [expr { $dstDelta / 3600 }]]
	    set mm [::format %02d [expr { ($dstDelta / 60 ) % 60 }]]
	    set ss [::format %02d [expr { $dstDelta % 60 }]]
	    append tzname < $dispDstSignum $hh $mm > $dstSignum $hh : $mm : $ss
	    if { $dstYear == 0 } {
		append tzname ,M $dstMonth . $dstDayOfMonth . $dstDayOfWeek
	    } else {
		# I have not been able to find any locale on which Windows
		# converts time zone on a fixed day of the year, hence don't
		# know how to interpret the fields.  If someone can inform me,
		# I'd be glad to code it up.  For right now, we bail out in
		# such a case.
		return :localtime
	    }
	    append tzname / [::format %02d $dstHour] \
		: [::format %02d $dstMinute] \
		: [::format %02d $dstSecond]
	    if { $stdYear == 0 } {
		append tzname ,M $stdMonth . $stdDayOfMonth . $stdDayOfWeek
	    } else {
		# I have not been able to find any locale on which Windows
		# converts time zone on a fixed day of the year, hence don't
		# know how to interpret the fields.  If someone can inform me,
		# I'd be glad to code it up.  For right now, we bail out in
		# such a case.
		return :localtime
	    }
	    append tzname / [::format %02d $stdHour] \
		: [::format %02d $stdMinute] \
		: [::format %02d $stdSecond]
	}
	dict set WinZoneInfo $data $tzname
    }

    return [dict get $WinZoneInfo $data]
}

#----------------------------------------------------------------------
#
# LoadTimeZoneFile --
#
#	Load the data file that specifies the conversion between a
#	given time zone and Greenwich.
#
# Parameters:
#	fileName -- Name of the file to load
#
# Results:
#	None.
#
# Side effects:
#	TZData(:fileName) contains the time zone data
#
#----------------------------------------------------------------------

proc ::tcl::clock::LoadTimeZoneFile { fileName } {
    variable DataDir
    variable TZData

    if { [info exists TZData($fileName)] } {
	return
    }

    # Since an unsafe interp uses the [clock] command in the parent, this code
    # is security sensitive.  Make sure that the path name cannot escape the
    # given directory.

    if { [regexp {^[/\\]|^[a-zA-Z]+:|(?:^|[/\\])\.\.} $fileName] } {
	return -code error \
	    -errorcode [list CLOCK badTimeZone :$fileName] \
	    "time zone \":$fileName\" not valid"
    }
    try {
	source [file join $DataDir $fileName]
    } on error {} {
	return -code error \
	    -errorcode [list CLOCK badTimeZone :$fileName] \
	    "time zone \":$fileName\" not found"
    }
    return
}

#----------------------------------------------------------------------
#
# LoadZoneinfoFile --
#
#	Loads a binary time zone information file in Olson format.
#
# Parameters:
#	fileName - Relative path name of the file to load.
#
# Results:
#	Returns an empty result normally; returns an error if no Olson file
#	was found or the file was malformed in some way.
#
# Side effects:
#	TZData(:fileName) contains the time zone data
#
#----------------------------------------------------------------------

proc ::tcl::clock::LoadZoneinfoFile { fileName } {
    variable ZoneinfoPaths

    # Since an unsafe interp uses the [clock] command in the parent, this code
    # is security sensitive.  Make sure that the path name cannot escape the
    # given directory.

    if { [regexp {^[/\\]|^[a-zA-Z]+:|(?:^|[/\\])\.\.} $fileName] } {
	return -code error \
	    -errorcode [list CLOCK badTimeZone :$fileName] \
	    "time zone \":$fileName\" not valid"
    }
    set fname ""
    foreach d $ZoneinfoPaths {
	set fname [file join $d $fileName]
	if { [file readable $fname] && [file isfile $fname] } {
	    break
	}
	set fname ""
    }
    if {$fname eq ""} {
	return -code error \
	    -errorcode [list CLOCK badTimeZone :$fileName] \
	    "time zone \":$fileName\" not found"
    }
    ReadZoneinfoFile $fileName $fname
}

#----------------------------------------------------------------------
#
# ReadZoneinfoFile --
#
#	Loads a binary time zone information file in Olson format.
#
# Parameters:
#	fileName - Name of the time zone (relative path name of the
#		   file).
#	fname - Absolute path name of the file.
#
# Results:
#	Returns an empty result normally; returns an error if no Olson file
#	was found or the file was malformed in some way.
#
# Side effects:
#	TZData(:fileName) contains the time zone data
#
#----------------------------------------------------------------------

proc ::tcl::clock::ReadZoneinfoFile {fileName fname} {
    variable MINWIDE
    variable TZData
    if { ![file exists $fname] } {
	return -code error "$fileName not found"
    }

    if { [file size $fname] > 262144 } {
	return -code error "$fileName too big"
    }

    # Suck in all the data from the file

    set f [open $fname r]
    fconfigure $f -translation binary
    set d [read $f]
    close $f

    # The file begins with a magic number, sixteen reserved bytes, and then
    # six 4-byte integers giving counts of fields in the file.

    binary scan $d a4a1x15IIIIII \
	magic version nIsGMT nIsStd nLeap nTime nType nChar
    set seek 44
    set ilen 4
    set iformat I
    if { $magic != {TZif} } {
	return -code error "$fileName not a time zone information file"
    }
    if { $nType > 255 } {
	return -code error "$fileName contains too many time types"
    }
    # Accept only Posix-style zoneinfo.  Sorry, 'leaps' bigots.
    if { $nLeap != 0 } {
	return -code error "$fileName contains leap seconds"
    }

    # In a version 2 file, we use the second part of the file, which contains
    # 64-bit transition times.

    if {$version eq "2"} {
	set seek [expr {
	    44
	    + 5 * $nTime
	    + 6 * $nType
	    + 4 * $nLeap
	    + $nIsStd
	    + $nIsGMT
	    + $nChar
	}]
	binary scan $d @${seek}a4a1x15IIIIII \
	    magic version nIsGMT nIsStd nLeap nTime nType nChar
	if {$magic ne {TZif}} {
	    return -code error "seek address $seek miscomputed, magic = $magic"
	}
	set iformat W
	set ilen 8
	incr seek 44
    }

    # Next come ${nTime} transition times, followed by ${nTime} time type
    # codes.  The type codes are unsigned 1-byte quantities.  We insert an
    # arbitrary start time in front of the transitions.

    binary scan $d @${seek}${iformat}${nTime}c${nTime} times tempCodes
    incr seek [expr { ($ilen + 1) * $nTime }]
    set times [linsert $times 0 $MINWIDE]
    set codes {}
    foreach c $tempCodes {
	lappend codes [expr { $c & 0xFF }]
    }
    set codes [linsert $codes 0 0]

    # Next come ${nType} time type descriptions, each of which has an offset
    # (seconds east of GMT), a DST indicator, and an index into the
    # abbreviation text.

    for { set i 0 } { $i < $nType } { incr i } {
	binary scan $d @${seek}Icc gmtOff isDst abbrInd
	lappend types [list $gmtOff $isDst $abbrInd]
	incr seek 6
    }

    # Next come $nChar characters of time zone name abbreviations, which are
    # null-terminated.
    # We build them up into a dictionary indexed by character index, because
    # that's what's in the indices above.

    binary scan $d @${seek}a${nChar} abbrs
    incr seek ${nChar}
    set abbrList [split $abbrs \0]
    set i 0
    set abbrevs {}
    foreach a $abbrList {
	for {set j 0} {$j <= [string length $a]} {incr j} {
	    dict set abbrevs $i [string range $a $j end]
	    incr i
	}
    }

    # Package up a list of tuples, each of which contains transition time,
    # seconds east of Greenwich, DST flag and time zone abbreviation.

    set r {}
    set lastTime $MINWIDE
    foreach t $times c $codes {
	if { $t < $lastTime } {
	    return -code error "$fileName has times out of order"
	}
	set lastTime $t
	lassign [lindex $types $c] gmtoff isDst abbrInd
	set abbrev [dict get $abbrevs $abbrInd]
	lappend r [list $t $gmtoff $isDst $abbrev]
    }

    # In a version 2 file, there is also a POSIX-style time zone description
    # at the very end of the file.  To get to it, skip over nLeap leap second
    # values (8 bytes each),
    # nIsStd standard/DST indicators and nIsGMT UTC/local indicators.

    if {$version eq {2}} {
	set seek [expr {$seek + 8 * $nLeap + $nIsStd + $nIsGMT + 1}]
	set last [string first \n $d $seek]
	set posix [string range $d $seek [expr {$last-1}]]
	if {[llength $posix] > 0} {
	    set posixFields [ParsePosixTimeZone $posix]
	    foreach tuple [ProcessPosixTimeZone $posixFields] {
		lassign $tuple t gmtoff isDst abbrev
		if {$t > $lastTime} {
		    lappend r $tuple
		}
	    }
	}
    }

    set TZData(:$fileName) $r

    return
}

#----------------------------------------------------------------------
#
# ParsePosixTimeZone --
#
#	Parses the TZ environment variable in Posix form
#
# Parameters:
#	tz	Time zone specifier to be interpreted
#
# Results:
#	Returns a dictionary whose values contain the various pieces of the
#	time zone specification.
#
# Side effects:
#	None.
#
# Errors:
#	Throws an error if the syntax of the time zone is incorrect.
#
# The following keys are present in the dictionary:
#	stdName - Name of the time zone when Daylight Saving Time
#		  is not in effect.
#	stdSignum - Sign (+, -, or empty) of the offset from Greenwich
#		    to the given (non-DST) time zone.  + and the empty
#		    string denote zones west of Greenwich, - denotes east
#		    of Greenwich; this is contrary to the ISO convention
#		    but follows Posix.
#	stdHours - Hours part of the offset from Greenwich to the given
#		   (non-DST) time zone.
#	stdMinutes - Minutes part of the offset from Greenwich to the
#		     given (non-DST) time zone. Empty denotes zero.
#	stdSeconds - Seconds part of the offset from Greenwich to the
#		     given (non-DST) time zone. Empty denotes zero.
#	dstName - Name of the time zone when DST is in effect, or the
#		  empty string if the time zone does not observe Daylight
#		  Saving Time.
#	dstSignum, dstHours, dstMinutes, dstSeconds -
#		Fields corresponding to stdSignum, stdHours, stdMinutes,
#		stdSeconds for the Daylight Saving Time version of the
#		time zone.  If dstHours is empty, it is presumed to be 1.
#	startDayOfYear - The ordinal number of the day of the year on which
#			 Daylight Saving Time begins.  If this field is
#			 empty, then DST begins on a given month-week-day,
#			 as below.
#	startJ - The letter J, or an empty string.  If a J is present in
#		 this field, then startDayOfYear does not count February 29
#		 even in leap years.
#	startMonth - The number of the month in which Daylight Saving Time
#		     begins, supplied if startDayOfYear is empty.  If both
#		     startDayOfYear and startMonth are empty, then US rules
#		     are presumed.
#	startWeekOfMonth - The number of the week in the month in which
#			   Daylight Saving Time begins, in the range 1-5.
#			   5 denotes the last week of the month even in a
#			   4-week month.
#	startDayOfWeek - The number of the day of the week (Sunday=0,
#			 Saturday=6) on which Daylight Saving Time begins.
#	startHours - The hours part of the time of day at which Daylight
#		     Saving Time begins. An empty string is presumed to be 2.
#	startMinutes - The minutes part of the time of day at which DST begins.
#		       An empty string is presumed zero.
#	startSeconds - The seconds part of the time of day at which DST begins.
#		       An empty string is presumed zero.
#	endDayOfYear, endJ, endMonth, endWeekOfMonth, endDayOfWeek,
#	endHours, endMinutes, endSeconds -
#		Specify the end of DST in the same way that the start* fields
#		specify the beginning of DST.
#
# This procedure serves only to break the time specifier into fields.  No
# attempt is made to canonicalize the fields or supply default values.
#
#----------------------------------------------------------------------

proc ::tcl::clock::ParsePosixTimeZone { tz } {
    if {[regexp -expanded -nocase -- {
	^
	# 1 - Standard time zone name
	([[:alpha:]]+ | <[-+[:alnum:]]+>)
	# 2 - Standard time zone offset, signum
	([-+]?)
	# 3 - Standard time zone offset, hours
	([[:digit:]]{1,2})
	(?:
	    # 4 - Standard time zone offset, minutes
	    : ([[:digit:]]{1,2})
	    (?:
		# 5 - Standard time zone offset, seconds
		: ([[:digit:]]{1,2} )
	    )?
	)?
	(?:
	    # 6 - DST time zone name
	    ([[:alpha:]]+ | <[-+[:alnum:]]+>)
	    (?:
		(?:
		    # 7 - DST time zone offset, signum
		    ([-+]?)
		    # 8 - DST time zone offset, hours
		    ([[:digit:]]{1,2})
		    (?:
			# 9 - DST time zone offset, minutes
			: ([[:digit:]]{1,2})
			(?:
			    # 10 - DST time zone offset, seconds
			    : ([[:digit:]]{1,2})
			)?
		    )?
		)?
		(?:
		    ,
		    (?:
			# 11 - Optional J in n and Jn form 12 - Day of year
			( J ? )	( [[:digit:]]+ )
			| M
			# 13 - Month number 14 - Week of month 15 - Day of week
			( [[:digit:]] + )
			[.] ( [[:digit:]] + )
			[.] ( [[:digit:]] + )
		    )
		    (?:
			# 16 - Start time of DST - hours
			/ ( [[:digit:]]{1,2} )
			(?:
			    # 17 - Start time of DST - minutes
			    : ( [[:digit:]]{1,2} )
			    (?:
				# 18 - Start time of DST - seconds
				: ( [[:digit:]]{1,2} )
			    )?
			)?
		    )?
		    ,
		    (?:
			# 19 - Optional J in n and Jn form 20 - Day of year
			( J ? )	( [[:digit:]]+ )
			| M
			# 21 - Month number 22 - Week of month 23 - Day of week
			( [[:digit:]] + )
			[.] ( [[:digit:]] + )
			[.] ( [[:digit:]] + )
		    )
		    (?:
			# 24 - End time of DST - hours
			/ ( [[:digit:]]{1,2} )
			(?:
			    # 25 - End time of DST - minutes
			    : ( [[:digit:]]{1,2} )
			    (?:
				# 26 - End time of DST - seconds
				: ( [[:digit:]]{1,2} )
			    )?
			)?
		    )?
		)?
	    )?
	)?
	$
    } $tz -> x(stdName) x(stdSignum) x(stdHours) x(stdMinutes) x(stdSeconds) \
	     x(dstName) x(dstSignum) x(dstHours) x(dstMinutes) x(dstSeconds) \
	     x(startJ) x(startDayOfYear) \
	     x(startMonth) x(startWeekOfMonth) x(startDayOfWeek) \
	     x(startHours) x(startMinutes) x(startSeconds) \
	     x(endJ) x(endDayOfYear) \
	     x(endMonth) x(endWeekOfMonth) x(endDayOfWeek) \
	     x(endHours) x(endMinutes) x(endSeconds)] } {
	# it's a good timezone

	return [array get x]
    }

    return -code error\
	-errorcode [list CLOCK badTimeZone $tz] \
	"unable to parse time zone specification \"$tz\""
}

#----------------------------------------------------------------------
#
# ProcessPosixTimeZone --
#
#	Handle a Posix time zone after it's been broken out into fields.
#
# Parameters:
#	z - Dictionary returned from 'ParsePosixTimeZone'
#
# Results:
#	Returns time zone information for the 'TZData' array.
#
# Side effects:
#	None.
#
#----------------------------------------------------------------------

proc ::tcl::clock::ProcessPosixTimeZone { z } {
    variable MINWIDE
    variable TZData

    # Determine the standard time zone name and seconds east of Greenwich

    set stdName [dict get $z stdName]
    if { [string index $stdName 0] eq {<} } {
	set stdName [string range $stdName 1 end-1]
    }
    if { [dict get $z stdSignum] eq {-} } {
	set stdSignum +1
    } else {
	set stdSignum -1
    }
    set stdHours [lindex [::scan [dict get $z stdHours] %d] 0]
    if { [dict get $z stdMinutes] ne {} } {
	set stdMinutes [lindex [::scan [dict get $z stdMinutes] %d] 0]
    } else {
	set stdMinutes 0
    }
    if { [dict get $z stdSeconds] ne {} } {
	set stdSeconds [lindex [::scan [dict get $z stdSeconds] %d] 0]
    } else {
	set stdSeconds 0
    }
    set stdOffset [expr {
	(($stdHours * 60 + $stdMinutes) * 60 + $stdSeconds) * $stdSignum
    }]
    set data [list [list $MINWIDE $stdOffset 0 $stdName]]

    # If there's no daylight zone, we're done

    set dstName [dict get $z dstName]
    if { $dstName eq {} } {
	return $data
    }
    if { [string index $dstName 0] eq {<} } {
	set dstName [string range $dstName 1 end-1]
    }

    # Determine the daylight name

    if { [dict get $z dstSignum] eq {-} } {
	set dstSignum +1
    } else {
	set dstSignum -1
    }
    if { [dict get $z dstHours] eq {} } {
	set dstOffset [expr { 3600 + $stdOffset }]
    } else {
	set dstHours [lindex [::scan [dict get $z dstHours] %d] 0]
	if { [dict get $z dstMinutes] ne {} } {
	    set dstMinutes [lindex [::scan [dict get $z dstMinutes] %d] 0]
	} else {
	    set dstMinutes 0
	}
	if { [dict get $z dstSeconds] ne {} } {
	    set dstSeconds [lindex [::scan [dict get $z dstSeconds] %d] 0]
	} else {
	    set dstSeconds 0
	}
	set dstOffset [expr {
	    (($dstHours*60 + $dstMinutes) * 60 + $dstSeconds) * $dstSignum
	}]
    }

    # Fill in defaults for European or US DST rules
    # US start time is the second Sunday in March
    # EU start time is the last Sunday in March
    # US end time is the first Sunday in November.
    # EU end time is the last Sunday in October

    if {
	[dict get $z startDayOfYear] eq {}
	&& [dict get $z startMonth] eq {}
    } then {
	if {($stdSignum * $stdHours>=0) && ($stdSignum * $stdHours<=12)} {
	    # EU
	    dict set z startWeekOfMonth 5
	    if {$stdHours>2} {
		dict set z startHours 2
	    } else {
		dict set z startHours [expr {$stdHours+1}]
	    }
	} else {
	    # US
	    dict set z startWeekOfMonth 2
	    dict set z startHours 2
	}
	dict set z startMonth 3
	dict set z startDayOfWeek 0
	dict set z startMinutes 0
	dict set z startSeconds 0
    }
    if {
	[dict get $z endDayOfYear] eq {}
	&& [dict get $z endMonth] eq {}
    } then {
	if {($stdSignum * $stdHours>=0) && ($stdSignum * $stdHours<=12)} {
	    # EU
	    dict set z endMonth 10
	    dict set z endWeekOfMonth 5
	    if {$stdHours>2} {
		dict set z endHours 3
	    } else {
		dict set z endHours [expr {$stdHours+2}]
	    }
	} else {
	    # US
	    dict set z endMonth 11
	    dict set z endWeekOfMonth 1
	    dict set z endHours 2
	}
	dict set z endDayOfWeek 0
	dict set z endMinutes 0
	dict set z endSeconds 0
    }

    # Put DST in effect in all years from 1916 to 2099.

    for { set y 1916 } { $y < 2100 } { incr y } {
	set startTime [DeterminePosixDSTTime $z start $y]
	incr startTime [expr { - wide($stdOffset) }]
	set endTime [DeterminePosixDSTTime $z end $y]
	incr endTime [expr { - wide($dstOffset) }]
	if { $startTime < $endTime } {
	    lappend data \
		[list $startTime $dstOffset 1 $dstName] \
		[list $endTime $stdOffset 0 $stdName]
	} else {
	    lappend data \
		[list $endTime $stdOffset 0 $stdName] \
		[list $startTime $dstOffset 1 $dstName]
	}
    }

    return $data
}

#----------------------------------------------------------------------
#
# DeterminePosixDSTTime --
#
#	Determines the time that Daylight Saving Time starts or ends from a
#	Posix time zone specification.
#
# Parameters:
#	z - Time zone data returned from ParsePosixTimeZone.
#	    Missing fields are expected to be filled in with
#	    default values.
#	bound - The word 'start' or 'end'
#	y - The year for which the transition time is to be determined.
#
# Results:
#	Returns the transition time as a count of seconds from the epoch.  The
#	time is relative to the wall clock, not UTC.
#
#----------------------------------------------------------------------

proc ::tcl::clock::DeterminePosixDSTTime { z bound y } {

    variable FEB_28

    # Determine the start or end day of DST

    set date [dict create era CE year $y gregorian 1]
    set doy [dict get $z ${bound}DayOfYear]
    if { $doy ne {} } {

	# Time was specified as a day of the year

	if { [dict get $z ${bound}J] ne {}
	     && [IsGregorianLeapYear $date]
	     && ( $doy > $FEB_28 ) } {
	    incr doy
	}
	dict set date dayOfYear $doy
	set date [GetJulianDayFromEraYearDay $date[set date {}] 2361222]
    } else {
	# Time was specified as a day of the week within a month

	dict set date month [dict get $z ${bound}Month]
	dict set date dayOfWeek [dict get $z ${bound}DayOfWeek]
	set dowim [dict get $z ${bound}WeekOfMonth]
	if { $dowim >= 5 } {
	    set dowim -1
	}
	dict set date dayOfWeekInMonth $dowim
	set date [GetJulianDayFromEraYearMonthWeekDay $date[set date {}] 2361222]

    }

    set jd [dict get $date julianDay]
    set seconds [expr {
	wide($jd) * wide(86400) - wide(210866803200)
    }]

    set h [dict get $z ${bound}Hours]
    if { $h eq {} } {
	set h 2
    } else {
	set h [lindex [::scan $h %d] 0]
    }
    set m [dict get $z ${bound}Minutes]
    if { $m eq {} } {
	set m 0
    } else {
	set m [lindex [::scan $m %d] 0]
    }
    set s [dict get $z ${bound}Seconds]
    if { $s eq {} } {
	set s 0
    } else {
	set s [lindex [::scan $s %d] 0]
    }
    set tod [expr { ( $h * 60 + $m ) * 60 + $s }]
    return [expr { $seconds + $tod }]
}

#----------------------------------------------------------------------
#
# GetJulianDayFromEraYearDay --
#
#	Given a year, month and day on the Gregorian calendar, determines
#	the Julian Day Number beginning at noon on that date.
#
# Parameters:
#	date -- A dictionary in which the 'era', 'year', and
#		'dayOfYear' slots are populated. The calendar in use
#		is determined by the date itself relative to:
#       changeover -- Julian day on which the Gregorian calendar was
#		adopted in the current locale.
#
# Results:
#	Returns the given dictionary augmented with a 'julianDay' key whose
#	value is the desired Julian Day Number, and a 'gregorian' key that
#	specifies whether the calendar is Gregorian (1) or Julian (0).
#
# Side effects:
#	None.
#
# Bugs:
#	This code needs to be moved to the C layer.
#
#----------------------------------------------------------------------

proc ::tcl::clock::GetJulianDayFromEraYearDay {date changeover} {
    # Get absolute year number from the civil year

    switch -exact -- [dict get $date era] {
	BCE {
	    set year [expr { 1 - [dict get $date year] }]
	}
	CE {
	    set year [dict get $date year]
	}
    }
    set ym1 [expr { $year - 1 }]

    # Try the Gregorian calendar first.

    dict set date gregorian 1
    set jd [expr {
	1721425
	+ [dict get $date dayOfYear]
	+ ( 365 * $ym1 )
	+ ( $ym1 / 4 )
	- ( $ym1 / 100 )
	+ ( $ym1 / 400 )
    }]

    # If the date is before the Gregorian change, use the Julian calendar.

    if { $jd < $changeover } {
	dict set date gregorian 0
	set jd [expr {
	    1721423
	    + [dict get $date dayOfYear]
	    + ( 365 * $ym1 )
	    + ( $ym1 / 4 )
	}]
    }

    dict set date julianDay $jd
    return $date
}

#----------------------------------------------------------------------
#
# GetJulianDayFromEraYearMonthWeekDay --
#
#	Determines the Julian Day number corresponding to the nth given
#	day-of-the-week in a given month.
#
# Parameters:
#	date - Dictionary containing the keys, 'era', 'year', 'month'
#	       'weekOfMonth', 'dayOfWeek', and 'dayOfWeekInMonth'.
#	changeover - Julian Day of adoption of the Gregorian calendar
#
# Results:
#	Returns the given dictionary, augmented with a 'julianDay' key.
#
# Side effects:
#	None.
#
# Bugs:
#	This code needs to be moved to the C layer.
#
#----------------------------------------------------------------------

proc ::tcl::clock::GetJulianDayFromEraYearMonthWeekDay {date changeover} {
    # Come up with a reference day; either the zeroeth day of the given month
    # (dayOfWeekInMonth >= 0) or the seventh day of the following month
    # (dayOfWeekInMonth < 0)

    set date2 $date
    set week [dict get $date dayOfWeekInMonth]
    if { $week >= 0 } {
	dict set date2 dayOfMonth 0
    } else {
	dict incr date2 month
	dict set date2 dayOfMonth 7
    }
    set date2 [GetJulianDayFromEraYearMonthDay $date2[set date2 {}] \
		   $changeover]
    set wd0 [WeekdayOnOrBefore [dict get $date dayOfWeek] \
		 [dict get $date2 julianDay]]
    dict set date julianDay [expr { $wd0 + 7 * $week }]
    return $date
}

#----------------------------------------------------------------------
#
# IsGregorianLeapYear --
#
#	Determines whether a given date represents a leap year in the
#	Gregorian calendar.
#
# Parameters:
#	date -- The date to test.  The fields, 'era', 'year' and 'gregorian'
#	        must be set.
#
# Results:
#	Returns 1 if the year is a leap year, 0 otherwise.
#
# Side effects:
#	None.
#
#----------------------------------------------------------------------

proc ::tcl::clock::IsGregorianLeapYear { date } {
    switch -exact -- [dict get $date era] {
	BCE {
	    set year [expr { 1 - [dict get $date year]}]
	}
	CE {
	    set year [dict get $date year]
	}
    }
    if { $year % 4 != 0 } {
	return 0
    } elseif { ![dict get $date gregorian] } {
	return 1
    } elseif { $year % 400 == 0 } {
	return 1
    } elseif { $year % 100 == 0 } {
	return 0
    } else {
	return 1
    }
}

#----------------------------------------------------------------------
#
# WeekdayOnOrBefore --
#
#	Determine the nearest day of week (given by the 'weekday' parameter,
#	Sunday==0) on or before a given Julian Day.
#
# Parameters:
#	weekday -- Day of the week
#	j -- Julian Day number
#
# Results:
#	Returns the Julian Day Number of the desired date.
#
# Side effects:
#	None.
#
#----------------------------------------------------------------------

proc ::tcl::clock::WeekdayOnOrBefore { weekday j } {
    set k [expr { ( $weekday + 6 )  % 7 }]
    return [expr { $j - ( $j - $k ) % 7 }]
}

#----------------------------------------------------------------------
#
# ChangeCurrentLocale --
#
#        The global locale was changed within msgcat.
#        Clears the buffered parse functions of the current locale.
#
# Parameters:
#        loclist (ignored)
#
# Results:
#        None.
#
# Side effects:
#        Buffered parse functions are cleared.
#
#----------------------------------------------------------------------

proc ::tcl::clock::ChangeCurrentLocale {args} {
    ::tcl::unsupported::clock::configure -current-locale [lindex $args 0]
}

#----------------------------------------------------------------------
#
# ClearCaches --
#
#	Clears all caches to reclaim the memory used in [clock]
#
# Parameters:
#	None.
#
# Results:
#	None.
#
# Side effects:
#	Caches are cleared.
#
#----------------------------------------------------------------------

proc ::tcl::clock::ClearCaches {} {
    variable LocFmtMap
    variable mcMergedCat
    variable TimeZoneBad

    # tell backend - should invalidate:
    ::tcl::unsupported::clock::configure -clear

    # clear msgcat cache:
    set mcMergedCat [dict create]

    set LocFmtMap {}
    set TimeZoneBad {}
    InitTZData
}
