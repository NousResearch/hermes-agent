"""Wrapper to the POSIX crypt library call and associated functionality."""

import sys as _sys

try:
    import _crypt
except ModuleNotFoundError:
    if _sys.platform == 'win32':
        raise ImportError("The crypt module is not supported on Windows")
    else:
        raise ImportError("The required _crypt module was not built as part of CPython")

import errno
import string as _string
import warnings
from random import SystemRandom as _SystemRandom
from collections import namedtuple as _namedtuple


warnings._deprecated(__name__, remove=(3, 13))


_saltchars = _string.ascii_letters + _string.digits + './'
_sr = _SystemRandom()


class _Method(_namedtuple('_Method', 'name ident salt_chars total_size')):

    """Class representing a salt method per the Modular Crypt Format or the
    legacy 2-character crypt method."""

    def __repr__(self):
        return '<crypt.METHOD_{}>'.format(self.name)


def mksalt(method=None, *, rounds=None):
    """Generate a salt for the specified method.

    If not specified, the strongest available method will be used.

    """
    if method is None:
        method = methods[0]
    if rounds is not None and not isinstance(rounds, int):
        raise TypeError(f'{rounds.__class__.__name__} object cannot be '
                        f'interpreted as an integer')
    if not method.ident:  # traditional
        s = ''
    else:  # modular
        s = f'${method.ident}$'

    if method.ident and method.ident[0] == '2':  # Blowfish variants
        if rounds is None:
            log_rounds = 12
        else:
            log_rounds = int.bit_length(rounds-1)
            if rounds != 1 << log_rounds:
                raise ValueError('rounds must be a power of 2')
            if not 4 <= log_rounds <= 31:
                raise ValueError('rounds out of the range 2**4 to 2**31')
        s += f'{log_rounds:02d}$'
    elif method.ident in ('5', '6'):  # SHA-2
        if rounds is not None:
            if not 1000 <= rounds <= 999_999_999:
                raise ValueError('rounds out of the range 1000 to 999_999_999')
            s += f'rounds={rounds}$'
    elif rounds is not None:
        raise ValueError(f"{method} doesn't support the rounds argument")

    s += ''.join(_sr.choice(_saltchars) for char in range(method.salt_chars))
    return s


def crypt(word, salt=None):
    """Return a string representing the one-way hash of a password, with a salt
    prepended.

    If ``salt`` is not specified or is ``None``, the strongest
    available method will be selected and a salt generated.  Otherwise,
    ``salt`` may be one of the ``crypt.METHOD_*`` values, or a string as
    returned by ``crypt.mksalt()``.

    """
    if salt is None or isinstance(salt, _Method):
        salt = mksalt(salt)
    return _crypt.crypt(word, salt)


#  available salting/crypto methods
methods = []

def _add_method(name, *args, rounds=None):
    method = _Method(name, *args)
    globals()['METHOD_' + name] = method
    salt = mksalt(method, rounds=rounds)
    result = None
    try:
        result = crypt('', salt)
    except OSError as e:
        # Not all libc libraries support all encryption methods.
        if e.errno in {errno.EINVAL, errno.EPERM, errno.ENOSYS}:
            return False
        raise
    if result and len(result) == method.total_size:
        methods.append(method)
        return True
    return False

_add_method('SHA512', '6', 16, 106)
_add_method('SHA256', '5', 16, 63)

# Choose the strongest supported version of Blowfish hashing.
# Early versions have flaws.  Version 'a' fixes flaws of
# the initial implementation, 'b' fixes flaws of 'a'.
# 'y' is the same as 'b', for compatibility
# with openwall crypt_blowfish.
for _v in 'b', 'y', 'a', '':
    if _add_method('BLOWFISH', '2' + _v, 22, 59 + len(_v), rounds=1<<4):
        break

_add_method('MD5', '1', 8, 34)
_add_method('CRYPT', None, 2, 13)

del _v, _add_method
