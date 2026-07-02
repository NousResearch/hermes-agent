import importlib
import sys


def bypass_compiler_fixup(cmd, args):
    return cmd


if sys.platform == 'darwin':
    compiler_fixup = importlib.import_module('_osx_support').compiler_fixup
else:
    compiler_fixup = bypass_compiler_fixup
