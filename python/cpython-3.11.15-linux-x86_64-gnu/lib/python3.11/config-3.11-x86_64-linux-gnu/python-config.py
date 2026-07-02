#!/bin/sh
'''exec' "$(dirname -- "$(realpath -- "$0")")/python3.11" "$0" "$@"
' '''
# -*- python -*-

# Keep this script in sync with python-config.sh.in

import getopt
import os
import sys
import sysconfig

valid_opts = ['prefix', 'exec-prefix', 'includes', 'libs', 'cflags',
              'ldflags', 'extension-suffix', 'help', 'abiflags', 'configdir',
              'embed']

def exit_with_usage(code=1):
    print("Usage: {0} [{1}]".format(
        sys.argv[0], '|'.join('--'+opt for opt in valid_opts)), file=sys.stderr)
    sys.exit(code)

try:
    opts, args = getopt.getopt(sys.argv[1:], '', valid_opts)
except getopt.error:
    exit_with_usage()

if not opts:
    exit_with_usage()

getvar = sysconfig.get_config_var
pyver = getvar('VERSION')

opt_flags = [flag for (flag, val) in opts]

if '--help' in opt_flags:
    exit_with_usage(code=0)

for opt in opt_flags:
    if opt == '--prefix':
        print(getvar('prefix'))

    elif opt == '--exec-prefix':
        print(getvar('exec_prefix'))

    elif opt in ('--includes', '--cflags'):
        flags = ['-I' + sysconfig.get_path('include'),
                 '-I' + sysconfig.get_path('platinclude')]
        if opt == '--cflags':
            flags.extend(getvar('CFLAGS').split())
        print(' '.join(flags))

    elif opt in ('--libs', '--ldflags'):
        libs = []
        if '--embed' in opt_flags:
            libs.append('-lpython' + pyver + sys.abiflags)
        else:
            libpython = getvar('LIBPYTHON')
            if libpython:
                libs.append(libpython)
        libs.extend(getvar('LIBS').split() + getvar('SYSLIBS').split())

        # add the prefix/lib/pythonX.Y/config dir, but only if there is no
        # shared library in prefix/lib/.
        if opt == '--ldflags':
            if not getvar('Py_ENABLE_SHARED'):
                libs.insert(0, '-L' + getvar('LIBPL'))
        print(' '.join(libs))

    elif opt == '--extension-suffix':
        print(getvar('EXT_SUFFIX'))

    elif opt == '--abiflags':
        print(sys.abiflags)

    elif opt == '--configdir':
        print(getvar('LIBPL'))
