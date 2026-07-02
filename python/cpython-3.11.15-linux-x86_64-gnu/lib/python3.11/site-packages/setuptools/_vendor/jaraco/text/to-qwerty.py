import sys

from . import layouts

__name__ == '__main__' and layouts._translate_stream(sys.stdin, layouts.to_qwerty)
