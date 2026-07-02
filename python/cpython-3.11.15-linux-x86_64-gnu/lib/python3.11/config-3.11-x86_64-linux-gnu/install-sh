#!/bin/sh
# install - install a program, script, or datafile

scriptversion=2020-11-14.01; # UTC

# This originates from X11R5 (mit/util/scripts/install.sh), which was
# later released in X11R6 (xc/config/util/install.sh) with the
# following copyright and license.
#
# Copyright (C) 1994 X Consortium
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# X CONSORTIUM BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
# AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNEC-
# TION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Except as contained in this notice, the name of the X Consortium shall not
# be used in advertising or otherwise to promote the sale, use or other deal-
# ings in this Software without prior written authorization from the X Consor-
# tium.
#
#
# FSF changes to this file are in the public domain.
#
# Calling this script install-sh is preferred over install.sh, to prevent
# 'make' implicit rules from creating a file called install from it
# when there is no Makefile.
#
# This script is compatible with the BSD install script, but was written
# from scratch.

tab='	'
nl='
'
IFS=" $tab$nl"

# Set DOITPROG to "echo" to test this script.

doit=${DOITPROG-}
doit_exec=${doit:-exec}

# Put in absolute file names if you don't have them in your path;
# or use environment vars.

chgrpprog=${CHGRPPROG-chgrp}
chmodprog=${CHMODPROG-chmod}
chownprog=${CHOWNPROG-chown}
cmpprog=${CMPPROG-cmp}
cpprog=${CPPROG-cp}
mkdirprog=${MKDIRPROG-mkdir}
mvprog=${MVPROG-mv}
rmprog=${RMPROG-rm}
stripprog=${STRIPPROG-strip}

posix_mkdir=

# Desired mode of installed file.
mode=0755

# Create dirs (including intermediate dirs) using mode 755.
# This is like GNU 'install' as of coreutils 8.32 (2020).
mkdir_umask=22

backupsuffix=
chgrpcmd=
chmodcmd=$chmodprog
chowncmd=
mvcmd=$mvprog
rmcmd="$rmprog -f"
stripcmd=

src=
dst=
dir_arg=
dst_arg=

copy_on_change=false
is_target_a_directory=possibly

usage="\
Usage: $0 [OPTION]... [-T] SRCFILE DSTFILE
   or: $0 [OPTION]... SRCFILES... DIRECTORY
   or: $0 [OPTION]... -t DIRECTORY SRCFILES...
   or: $0 [OPTION]... -d DIRECTORIES...

In the 1st form, copy SRCFILE to DSTFILE.
In the 2nd and 3rd, copy all SRCFILES to DIRECTORY.
In the 4th, create DIRECTORIES.

Options:
     --help     display this help and exit.
     --version  display version info and exit.

  -c            (ignored)
  -C            install only if different (preserve data modification time)
  -d            create directories instead of installing files.
  -g GROUP      $chgrpprog installed files to GROUP.
  -m MODE       $chmodprog installed files to MODE.
  -o USER       $chownprog installed files to USER.
  -p            pass -p to $cpprog.
  -s            $stripprog installed files.
  -S SUFFIX     attempt to back up existing files, with suffix SUFFIX.
  -t DIRECTORY  install into DIRECTORY.
  -T            report an error if DSTFILE is a directory.

Environment variables override the default commands:
  CHGRPPROG CHMODPROG CHOWNPROG CMPPROG CPPROG MKDIRPROG MVPROG
  RMPROG STRIPPROG

By default, rm is invoked with -f; when overridden with RMPROG,
it's up to you to specify -f if you want it.

If -S is not specified, no backups are attempted.

Email bug reports to bug-automake@gnu.org.
Automake home page: https://www.gnu.org/software/automake/
"

while test $# -ne 0; do
  case $1 in
    -c) ;;

    -C) copy_on_change=true;;

    -d) dir_arg=true;;

    -g) chgrpcmd="$chgrpprog $2"
        shift;;

    --help) echo "$usage"; exit $?;;

    -m) mode=$2
        case $mode in
          *' '* | *"$tab"* | *"$nl"* | *'*'* | *'?'* | *'['*)
            echo "$0: invalid mode: $mode" >&2
            exit 1;;
        esac
        shift;;

    -o) chowncmd="$chownprog $2"
        shift;;

    -p) cpprog="$cpprog -p";;

    -s) stripcmd=$stripprog;;

    -S) backupsuffix="$2"
        shift;;

    -t)
        is_target_a_directory=always
        dst_arg=$2
        # Protect names problematic for 'test' and other utilities.
        case $dst_arg in
          -* | [=\(\)!]) dst_arg=./$dst_arg;;
        esac
        shift;;

    -T) is_target_a_directory=never;;

    --version) echo "$0 $scriptversion"; exit $?;;

    --) shift
        break;;

    -*) echo "$0: invalid option: $1" >&2
        exit 1;;

    *)  break;;
  esac
  shift
done

# We allow the use of options -d and -T together, by making -d
# take the precedence; this is for compatibility with GNU install.

if test -n "$dir_arg"; then
  if test -n "$dst_arg"; then
    echo "$0: target directory not allowed when installing a directory." >&2
    exit 1
  fi
fi

if test $# -ne 0 && test -z "$dir_arg$dst_arg"; then
  # When -d is used, all remaining arguments are directories to create.
  # When -t is used, the destination is already specified.
  # Otherwise, the last argument is the destination.  Remove it from $@.
  for arg
  do
    if test -n "$dst_arg"; then
      # $@ is not empty: it contains at least $arg.
      set fnord "$@" "$dst_arg"
      shift # fnord
    fi
    shift # arg
    dst_arg=$arg
    # Protect names problematic for 'test' and other utilities.
    case $dst_arg in
      -* | [=\(\)!]) dst_arg=./$dst_arg;;
    esac
  done
fi

if test $# -eq 0; then
  if test -z "$dir_arg"; then
    echo "$0: no input file specified." >&2
    exit 1
  fi
  # It's OK to call 'install-sh -d' without argument.
  # This can happen when creating conditional directories.
  exit 0
fi

if test -z "$dir_arg"; then
  if test $# -gt 1 || test "$is_target_a_directory" = always; then
    if test ! -d "$dst_arg"; then
      echo "$0: $dst_arg: Is not a directory." >&2
      exit 1
    fi
  fi
fi

if test -z "$dir_arg"; then
  do_exit='(exit $ret); exit $ret'
  trap "ret=129; $do_exit" 1
  trap "ret=130; $do_exit" 2
  trap "ret=141; $do_exit" 13
  trap "ret=143; $do_exit" 15

  # Set umask so as not to create temps with too-generous modes.
  # However, 'strip' requires both read and write access to temps.
  case $mode in
    # Optimize common cases.
    *644) cp_umask=133;;
    *755) cp_umask=22;;

    *[0-7])
      if test -z "$stripcmd"; then
        u_plus_rw=
      else
        u_plus_rw='% 200'
      fi
      cp_umask=`expr '(' 777 - $mode % 1000 ')' $u_plus_rw`;;
    *)
      if test -z "$stripcmd"; then
        u_plus_rw=
      else
        u_plus_rw=,u+rw
      fi
      cp_umask=$mode$u_plus_rw;;
  esac
fi

for src
do
  # Protect names problematic for 'test' and other utilities.
  case $src in
    -* | [=\(\)!]) src=./$src;;
  esac

  if test -n "$dir_arg"; then
    dst=$src
    dstdir=$dst
    test -d "$dstdir"
    dstdir_status=$?
    # Don't chown directories that already exist.
    if test $dstdir_status = 0; then
      chowncmd=""
    fi
  else

    # Waiting for this to be detected by the "$cpprog $src $dsttmp" command
    # might cause directories to be created, which would be especially bad
    # if $src (and thus $dsttmp) contains '*'.
    if test ! -f "$src" && test ! -d "$src"; then
      echo "$0: $src does not exist." >&2
      exit 1
    fi

    if test -z "$dst_arg"; then
      echo "$0: no destination specified." >&2
      exit 1
    fi
    dst=$dst_arg

    # If destination is a directory, append the input filename.
    if test -d "$dst"; then
      if test "$is_target_a_directory" = never; then
        echo "$0: $dst_arg: Is a directory" >&2
        exit 1
      fi
      dstdir=$dst
      dstbase=`basename "$src"`
      case $dst in
	*/) dst=$dst$dstbase;;
	*)  dst=$dst/$dstbase;;
      esac
      dstdir_status=0
    else
      dstdir=`dirname "$dst"`
      test -d "$dstdir"
      dstdir_status=$?
    fi
  fi

  case $dstdir in
    */) dstdirslash=$dstdir;;
    *)  dstdirslash=$dstdir/;;
  esac

  obsolete_mkdir_used=false

  if test $dstdir_status != 0; then
    case $posix_mkdir in
      '')
        # With -d, create the new directory with the user-specified mode.
        # Otherwise, rely on $mkdir_umask.
        if test -n "$dir_arg"; then
          mkdir_mode=-m$mode
        else
          mkdir_mode=
        fi

        posix_mkdir=false
	# The $RANDOM variable is not portable (e.g., dash).  Use it
	# here however when possible just to lower collision chance.
	tmpdir=${TMPDIR-/tmp}/ins$RANDOM-$$

	trap '
	  ret=$?
	  rmdir "$tmpdir/a/b" "$tmpdir/a" "$tmpdir" 2>/dev/null
	  exit $ret
	' 0

	# Because "mkdir -p" follows existing symlinks and we likely work
	# directly in world-writeable /tmp, make sure that the '$tmpdir'
	# directory is successfully created first before we actually test
	# 'mkdir -p'.
	if (umask $mkdir_umask &&
	    $mkdirprog $mkdir_mode "$tmpdir" &&
	    exec $mkdirprog $mkdir_mode -p -- "$tmpdir/a/b") >/dev/null 2>&1
	then
	  if test -z "$dir_arg" || {
	       # Check for POSIX incompatibilities with -m.
	       # HP-UX 11.23 and IRIX 6.5 mkdir -m -p sets group- or
	       # other-writable bit of parent directory when it shouldn't.
	       # FreeBSD 6.1 mkdir -m -p sets mode of existing directory.
	       test_tmpdir="$tmpdir/a"
	       ls_ld_tmpdir=`ls -ld "$test_tmpdir"`
	       case $ls_ld_tmpdir in
		 d????-?r-*) different_mode=700;;
		 d????-?--*) different_mode=755;;
		 *) false;;
	       esac &&
	       $mkdirprog -m$different_mode -p -- "$test_tmpdir" && {
		 ls_ld_tmpdir_1=`ls -ld "$test_tmpdir"`
		 test "$ls_ld_tmpdir" = "$ls_ld_tmpdir_1"
	       }
	     }
	  then posix_mkdir=:
	  fi
	  rmdir "$tmpdir/a/b" "$tmpdir/a" "$tmpdir"
	else
	  # Remove any dirs left behind by ancient mkdir implementations.
	  rmdir ./$mkdir_mode ./-p ./-- "$tmpdir" 2>/dev/null
	fi
	trap '' 0;;
    esac

    if
      $posix_mkdir && (
        umask $mkdir_umask &&
        $doit_exec $mkdirprog $mkdir_mode -p -- "$dstdir"
      )
    then :
    else

      # mkdir does not conform to POSIX,
      # or it failed possibly due to a race condition.  Create the
      # directory the slow way, step by step, checking for races as we go.

      case $dstdir in
        /*) prefix='/';;
        [-=\(\)!]*) prefix='./';;
        *)  prefix='';;
      esac

      oIFS=$IFS
      IFS=/
      set -f
      set fnord $dstdir
      shift
      set +f
      IFS=$oIFS

      prefixes=

      for d
      do
        test X"$d" = X && continue

        prefix=$prefix$d
        if test -d "$prefix"; then
          prefixes=
        else
          if $posix_mkdir; then
            (umask $mkdir_umask &&
             $doit_exec $mkdirprog $mkdir_mode -p -- "$dstdir") && break
            # Don't fail if two instances are running concurrently.
            test -d "$prefix" || exit 1
          else
            case $prefix in
              *\'*) qprefix=`echo "$prefix" | sed "s/'/'\\\\\\\\''/g"`;;
              *) qprefix=$prefix;;
            esac
            prefixes="$prefixes '$qprefix'"
          fi
        fi
        prefix=$prefix/
      done

      if test -n "$prefixes"; then
        # Don't fail if two instances are running concurrently.
        (umask $mkdir_umask &&
         eval "\$doit_exec \$mkdirprog $prefixes") ||
          test -d "$dstdir" || exit 1
        obsolete_mkdir_used=true
      fi
    fi
  fi

  if test -n "$dir_arg"; then
    { test -z "$chowncmd" || $doit $chowncmd "$dst"; } &&
    { test -z "$chgrpcmd" || $doit $chgrpcmd "$dst"; } &&
    { test "$obsolete_mkdir_used$chowncmd$chgrpcmd" = false ||
      test -z "$chmodcmd" || $doit $chmodcmd $mode "$dst"; } || exit 1
  else

    # Make a couple of temp file names in the proper directory.
    dsttmp=${dstdirslash}_inst.$$_
    rmtmp=${dstdirslash}_rm.$$_

    # Trap to clean up those temp files at exit.
    trap 'ret=$?; rm -f "$dsttmp" "$rmtmp" && exit $ret' 0

    # Copy the file name to the temp name.
    (umask $cp_umask &&
     { test -z "$stripcmd" || {
	 # Create $dsttmp read-write so that cp doesn't create it read-only,
	 # which would cause strip to fail.
	 if test -z "$doit"; then
	   : >"$dsttmp" # No need to fork-exec 'touch'.
	 else
	   $doit touch "$dsttmp"
	 fi
       }
     } &&
     $doit_exec $cpprog "$src" "$dsttmp") &&

    # and set any options; do chmod last to preserve setuid bits.
    #
    # If any of these fail, we abort the whole thing.  If we want to
    # ignore errors from any of these, just make sure not to ignore
    # errors from the above "$doit $cpprog $src $dsttmp" command.
    #
    { test -z "$chowncmd" || $doit $chowncmd "$dsttmp"; } &&
    { test -z "$chgrpcmd" || $doit $chgrpcmd "$dsttmp"; } &&
    { test -z "$stripcmd" || $doit $stripcmd "$dsttmp"; } &&
    { test -z "$chmodcmd" || $doit $chmodcmd $mode "$dsttmp"; } &&

    # If -C, don't bother to copy if it wouldn't change the file.
    if $copy_on_change &&
       old=`LC_ALL=C ls -dlL "$dst"     2>/dev/null` &&
       new=`LC_ALL=C ls -dlL "$dsttmp"  2>/dev/null` &&
       set -f &&
       set X $old && old=:$2:$4:$5:$6 &&
       set X $new && new=:$2:$4:$5:$6 &&
       set +f &&
       test "$old" = "$new" &&
       $cmpprog "$dst" "$dsttmp" >/dev/null 2>&1
    then
      rm -f "$dsttmp"
    else
      # If $backupsuffix is set, and the file being installed
      # already exists, attempt a backup.  Don't worry if it fails,
      # e.g., if mv doesn't support -f.
      if test -n "$backupsuffix" && test -f "$dst"; then
        $doit $mvcmd -f "$dst" "$dst$backupsuffix" 2>/dev/null
      fi

      # Rename the file to the real destination.
      $doit $mvcmd -f "$dsttmp" "$dst" 2>/dev/null ||

      # The rename failed, perhaps because mv can't rename something else
      # to itself, or perhaps because mv is so ancient that it does not
      # support -f.
      {
        # Now remove or move aside any old file at destination location.
        # We try this two ways since rm can't unlink itself on some
        # systems and the destination file might be busy for other
        # reasons.  In this case, the final cleanup might fail but the new
        # file should still install successfully.
        {
          test ! -f "$dst" ||
          $doit $rmcmd "$dst" 2>/dev/null ||
          { $doit $mvcmd -f "$dst" "$rmtmp" 2>/dev/null &&
            { $doit $rmcmd "$rmtmp" 2>/dev/null; :; }
          } ||
          { echo "$0: cannot unlink or rename $dst" >&2
            (exit 1); exit 1
          }
        } &&

        # Now rename the file to the real destination.
        $doit $mvcmd "$dsttmp" "$dst"
      }
    fi || exit 1

    trap '' 0
  fi
done

# Local variables:
# eval: (add-hook 'before-save-hook 'time-stamp)
# time-stamp-start: "scriptversion="
# time-stamp-format: "%:y-%02m-%02d.%02H"
# time-stamp-time-zone: "UTC0"
# time-stamp-end: "; # UTC"
# End:
