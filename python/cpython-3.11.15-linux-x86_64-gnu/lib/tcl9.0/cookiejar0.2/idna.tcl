# idna.tcl --
#
#	Implementation of IDNA (Internationalized Domain Names for
#	Applications) encoding/decoding system, built on a punycode engine
#	developed directly from the code in RFC 3492, Appendix C (with
#	substantial modifications).
#
# This implementation includes code from that RFC, translated to Tcl; the
# other parts are:
# Copyright © 2014 Donal K. Fellows
#
# See the file "license.terms" for information on usage and redistribution of
# this file, and for a DISCLAIMER OF ALL WARRANTIES.

namespace eval ::tcl::idna {
    namespace ensemble create -command puny -map {
	encode punyencode
	decode punydecode
    }
    namespace ensemble create -command ::tcl::idna -map {
	encode IDNAencode
	decode IDNAdecode
	puny puny
	version {::apply {{} {package present tcl::idna} ::}}
    }

    proc IDNAencode hostname {
	set parts {}
	# Split term from RFC 3490, Sec 3.1
	foreach part [split $hostname "\x2E\u3002\uFF0E\uFF61"] {
	    if {[regexp {[^-A-Za-z0-9]} $part]} {
		if {[regexp {[^-A-Za-z0-9\xA1-\uFFFF]} $part ch]} {
		    scan $ch %c c
		    if {$ch < "!" || $ch > "~"} {
			set ch [format "\\u%04x" $c]
		    }
		    throw [list IDNA INVALID_NAME_CHARACTER $ch] \
			"bad character \"$ch\" in DNS name"
		}
		set part xn--[punyencode $part]
		# Length restriction from RFC 5890, Sec 2.3.1
		if {[string length $part] > 63} {
		    throw [list IDNA OVERLONG_PART $part] \
			"hostname part too long"
		}
	    }
	    lappend parts $part
	}
	return [join $parts .]
    }
    proc IDNAdecode hostname {
	set parts {}
	# Split term from RFC 3490, Sec 3.1
	foreach part [split $hostname "\x2E\u3002\uFF0E\uFF61"] {
	    if {[string match -nocase "xn--*" $part]} {
		set part [punydecode [string range $part 4 end]]
	    }
	    lappend parts $part
	}
	return [join $parts .]
    }

    variable digits [split "abcdefghijklmnopqrstuvwxyz0123456789" ""]
    # Bootstring parameters for Punycode
    variable base 36
    variable tmin 1
    variable tmax 26
    variable skew 38
    variable damp 700
    variable initial_bias 72
    variable initial_n 0x80

    variable max_codepoint 0x10FFFF

    proc adapt {delta first numchars} {
	variable base
	variable tmin
	variable tmax
	variable damp
	variable skew

	set delta [expr {$delta / ($first ? $damp : 2)}]
	incr delta [expr {$delta / $numchars}]
	set k 0
	while {$delta > ($base - $tmin) * $tmax / 2} {
	    set delta [expr {$delta / ($base-$tmin)}]
	    incr k $base
	}
	return [expr {$k + ($base-$tmin+1) * $delta / ($delta+$skew)}]
    }

    # Main punycode encoding function
    proc punyencode {string {case ""}} {
	variable digits
	variable tmin
	variable tmax
	variable base
	variable initial_n
	variable initial_bias

	if {![string is boolean $case]} {
	    return -code error "\"$case\" must be boolean"
	}

	set in {}
	foreach char [set string [split $string ""]] {
	    scan $char "%c" ch
	    lappend in $ch
	}
	set output {}

	# Initialize the state:
	set n $initial_n
	set delta 0
	set bias $initial_bias

	# Handle the basic code points:
	foreach ch $string {
	    if {$ch < "\x80"} {
		if {$case eq ""} {
		    append output $ch
		} elseif {[string is true $case]} {
		    append output [string toupper $ch]
		} elseif {[string is false $case]} {
		    append output [string tolower $ch]
		}
	    }
	}

	set b [string length $output]

	# h is the number of code points that have been handled, b is the
	# number of basic code points.

	if {$b > 0} {
	    append output "-"
	}

	# Main encoding loop:

	for {set h $b} {$h < [llength $in]} {incr delta; incr n} {
	    # All non-basic code points < n have been handled already.  Find
	    # the next larger one:

	    set m inf
	    foreach ch $in {
		if {$ch >= $n && $ch < $m} {
		    set m $ch
		}
	    }

	    # Increase delta enough to advance the decoder's <n,i> state to
	    # <m,0>, but guard against overflow:

	    if {$m-$n > (0xFFFFFFFF-$delta)/($h+1)} {
		throw {PUNYCODE OVERFLOW} "overflow in delta computation"
	    }
	    incr delta [expr {($m-$n) * ($h+1)}]
	    set n $m

	    foreach ch $in {
		if {$ch < $n && ([incr delta] & 0xFFFFFFFF) == 0} {
		    throw {PUNYCODE OVERFLOW} "overflow in delta computation"
		}

		if {$ch != $n} {
		    continue
		}

		# Represent delta as a generalized variable-length integer:

		for {set q $delta; set k $base} true {incr k $base} {
		    set t [expr {min(max($k-$bias, $tmin), $tmax)}]
		    if {$q < $t} {
			break
		    }
		    append output \
			[lindex $digits [expr {$t + ($q-$t)%($base-$t)}]]
		    set q [expr {($q-$t) / ($base-$t)}]
		}

		append output [lindex $digits $q]
		set bias [adapt $delta [expr {$h==$b}] [expr {$h+1}]]
		set delta 0
		incr h
	    }
	}

	return $output
    }

    # Main punycode decode function
    proc punydecode {string {case ""}} {
	variable tmin
	variable tmax
	variable base
	variable initial_n
	variable initial_bias
	variable max_codepoint

	if {![string is boolean $case]} {
	    return -code error "\"$case\" must be boolean"
	}

	# Initialize the state:

	set n $initial_n
	set i 0
	set first 1
	set bias $initial_bias

	# Split the string into the "real" ASCII characters and the ones to
	# feed into the main decoder. Note that we don't need to check the
	# result of [regexp] because that RE will technically match any string
	# at all.

	regexp {^(?:(.*)-)?([^-]*)$} $string -> pre post
	if {[string is true -strict $case]} {
	    set pre [string toupper $pre]
	} elseif {[string is false -strict $case]} {
	    set pre [string tolower $pre]
	}
	set output [split $pre ""]
	set out [llength $output]

	# Main decoding loop:

	for {set in 0} {$in < [string length $post]} {incr in} {
	    # Decode a generalized variable-length integer into delta, which
	    # gets added to i. The overflow checking is easier if we increase
	    # i as we go, then subtract off its starting value at the end to
	    # obtain delta.

	    for {set oldi $i; set w 1; set k $base} 1 {incr in} {
		if {[set ch [string index $post $in]] eq ""} {
		    throw {PUNYCODE BAD_INPUT LENGTH} "exceeded input data"
		}
		if {[string match -nocase {[a-z]} $ch]} {
		    scan [string toupper $ch] %c digit
		    incr digit -65
		} elseif {[string match {[0-9]} $ch]} {
		    set digit [expr {$ch + 26}]
		} else {
		    throw {PUNYCODE BAD_INPUT CHAR} \
			    "bad decode character \"$ch\""
		}
		incr i [expr {$digit * $w}]
		set t [expr {min(max($tmin, $k-$bias), $tmax)}]
		if {$digit < $t} {
		    set bias [adapt [expr {$i-$oldi}] $first [incr out]]
		    set first 0
		    break
		}
		if {[set w [expr {$w * ($base - $t)}]] > 0x7FFFFFFF} {
		    throw {PUNYCODE OVERFLOW} \
			"excessively large integer computed in digit decode"
		}
		incr k $base
	    }

	    # i was supposed to wrap around from out+1 to 0, incrementing n
	    # each time, so we'll fix that now:

	    if {[incr n [expr {$i / $out}]] > 0x7FFFFFFF} {
		throw {PUNYCODE OVERFLOW} \
		    "excessively large integer computed in character choice"
	    } elseif {$n > $max_codepoint} {
		if {$n >= 0x00D800 && $n < 0x00E000} {
		    # Bare surrogate?!
		    throw {PUNYCODE NON_BMP} \
			[format "unsupported character U+%06x" $n]
		}
		throw {PUNYCODE NON_UNICODE} "bad codepoint $n"
	    }
	    set i [expr {$i % $out}]

	    # Insert n at position i of the output:

	    set output [linsert $output $i [format "%c" $n]]
	    incr i
	}

	return [join $output ""]
    }
}

package provide tcl::idna 1.0.1

# Local variables:
# mode: tcl
# fill-column: 78
# End:
