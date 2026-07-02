#! /usr/bin/env python3

# Copyright 1994 by Lance Ellinghouse
# Cathedral City, California Republic, United States of America.
#                        All Rights Reserved
# Permission to use, copy, modify, and distribute this software and its
# documentation for any purpose and without fee is hereby granted,
# provided that the above copyright notice appear in all copies and that
# both that copyright notice and this permission notice appear in
# supporting documentation, and that the name of Lance Ellinghouse
# not be used in advertising or publicity pertaining to distribution
# of the software without specific, written prior permission.
# LANCE ELLINGHOUSE DISCLAIMS ALL WARRANTIES WITH REGARD TO
# THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS, IN NO EVENT SHALL LANCE ELLINGHOUSE CENTRUM BE LIABLE
# FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
# OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#
# Modified by Jack Jansen, CWI, July 1995:
# - Use binascii module to do the actual line-by-line conversion
#   between ascii and binary. This results in a 1000-fold speedup. The C
#   version is still 5 times faster, though.
# - Arguments more compliant with python standard

"""Implementation of the UUencode and UUdecode functions.

encode(in_file, out_file [,name, mode], *, backtick=False)
decode(in_file [, out_file, mode, quiet])
"""

import binascii
import os
import sys
import warnings

warnings._deprecated(__name__, remove=(3, 13))

__all__ = ["Error", "encode", "decode"]

class Error(Exception):
    pass

def encode(in_file, out_file, name=None, mode=None, *, backtick=False):
    """Uuencode file"""
    #
    # If in_file is a pathname open it and change defaults
    #
    opened_files = []
    try:
        if in_file == '-':
            in_file = sys.stdin.buffer
        elif isinstance(in_file, str):
            if name is None:
                name = os.path.basename(in_file)
            if mode is None:
                try:
                    mode = os.stat(in_file).st_mode
                except AttributeError:
                    pass
            in_file = open(in_file, 'rb')
            opened_files.append(in_file)
        #
        # Open out_file if it is a pathname
        #
        if out_file == '-':
            out_file = sys.stdout.buffer
        elif isinstance(out_file, str):
            out_file = open(out_file, 'wb')
            opened_files.append(out_file)
        #
        # Set defaults for name and mode
        #
        if name is None:
            name = '-'
        if mode is None:
            mode = 0o666

        #
        # Remove newline chars from name
        #
        name = name.replace('\n','\\n')
        name = name.replace('\r','\\r')

        #
        # Write the data
        #
        out_file.write(('begin %o %s\n' % ((mode & 0o777), name)).encode("ascii"))
        data = in_file.read(45)
        while len(data) > 0:
            out_file.write(binascii.b2a_uu(data, backtick=backtick))
            data = in_file.read(45)
        if backtick:
            out_file.write(b'`\nend\n')
        else:
            out_file.write(b' \nend\n')
    finally:
        for f in opened_files:
            f.close()


def decode(in_file, out_file=None, mode=None, quiet=False):
    """Decode uuencoded file"""
    #
    # Open the input file, if needed.
    #
    opened_files = []
    if in_file == '-':
        in_file = sys.stdin.buffer
    elif isinstance(in_file, str):
        in_file = open(in_file, 'rb')
        opened_files.append(in_file)

    try:
        #
        # Read until a begin is encountered or we've exhausted the file
        #
        while True:
            hdr = in_file.readline()
            if not hdr:
                raise Error('No valid begin line found in input file')
            if not hdr.startswith(b'begin'):
                continue
            hdrfields = hdr.split(b' ', 2)
            if len(hdrfields) == 3 and hdrfields[0] == b'begin':
                try:
                    int(hdrfields[1], 8)
                    break
                except ValueError:
                    pass
        if out_file is None:
            # If the filename isn't ASCII, what's up with that?!?
            out_file = hdrfields[2].rstrip(b' \t\r\n\f').decode("ascii")
            if os.path.exists(out_file):
                raise Error(f'Cannot overwrite existing file: {out_file}')
            if (out_file.startswith(os.sep) or
                f'..{os.sep}' in out_file or (
                    os.altsep and
                    (out_file.startswith(os.altsep) or
                     f'..{os.altsep}' in out_file))
               ):
                raise Error(f'Refusing to write to {out_file} due to directory traversal')
        if mode is None:
            mode = int(hdrfields[1], 8)
        #
        # Open the output file
        #
        if out_file == '-':
            out_file = sys.stdout.buffer
        elif isinstance(out_file, str):
            fp = open(out_file, 'wb')
            os.chmod(out_file, mode)
            out_file = fp
            opened_files.append(out_file)
        #
        # Main decoding loop
        #
        s = in_file.readline()
        while s and s.strip(b' \t\r\n\f') != b'end':
            try:
                data = binascii.a2b_uu(s)
            except binascii.Error as v:
                # Workaround for broken uuencoders by /Fredrik Lundh
                nbytes = (((s[0]-32) & 63) * 4 + 5) // 3
                data = binascii.a2b_uu(s[:nbytes])
                if not quiet:
                    sys.stderr.write("Warning: %s\n" % v)
            out_file.write(data)
            s = in_file.readline()
        if not s:
            raise Error('Truncated input file')
    finally:
        for f in opened_files:
            f.close()

def test():
    """uuencode/uudecode main program"""

    import optparse
    parser = optparse.OptionParser(usage='usage: %prog [-d] [-t] [input [output]]')
    parser.add_option('-d', '--decode', dest='decode', help='Decode (instead of encode)?', default=False, action='store_true')
    parser.add_option('-t', '--text', dest='text', help='data is text, encoded format unix-compatible text?', default=False, action='store_true')

    (options, args) = parser.parse_args()
    if len(args) > 2:
        parser.error('incorrect number of arguments')
        sys.exit(1)

    # Use the binary streams underlying stdin/stdout
    input = sys.stdin.buffer
    output = sys.stdout.buffer
    if len(args) > 0:
        input = args[0]
    if len(args) > 1:
        output = args[1]

    if options.decode:
        if options.text:
            if isinstance(output, str):
                output = open(output, 'wb')
            else:
                print(sys.argv[0], ': cannot do -t to stdout')
                sys.exit(1)
        decode(input, output)
    else:
        if options.text:
            if isinstance(input, str):
                input = open(input, 'rb')
            else:
                print(sys.argv[0], ': cannot do -t from stdin')
                sys.exit(1)
        encode(input, output)

if __name__ == '__main__':
    test()
