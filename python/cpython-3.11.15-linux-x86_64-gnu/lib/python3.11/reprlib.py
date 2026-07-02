"""Redo the builtin repr() (representation) but with limits on most sizes."""

__all__ = ["Repr", "repr", "recursive_repr"]

import builtins
from itertools import islice
from _thread import get_ident

def recursive_repr(fillvalue='...'):
    'Decorator to make a repr function return fillvalue for a recursive call'

    def decorating_function(user_function):
        repr_running = set()

        def wrapper(self):
            key = id(self), get_ident()
            if key in repr_running:
                return fillvalue
            repr_running.add(key)
            try:
                result = user_function(self)
            finally:
                repr_running.discard(key)
            return result

        # Can't use functools.wraps() here because of bootstrap issues
        wrapper.__module__ = getattr(user_function, '__module__')
        wrapper.__doc__ = getattr(user_function, '__doc__')
        wrapper.__name__ = getattr(user_function, '__name__')
        wrapper.__qualname__ = getattr(user_function, '__qualname__')
        wrapper.__annotations__ = getattr(user_function, '__annotations__', {})
        return wrapper

    return decorating_function

class Repr:

    def __init__(self):
        self.fillvalue = '...'
        self.maxlevel = 6
        self.maxtuple = 6
        self.maxlist = 6
        self.maxarray = 5
        self.maxdict = 4
        self.maxset = 6
        self.maxfrozenset = 6
        self.maxdeque = 6
        self.maxstring = 30
        self.maxlong = 40
        self.maxother = 30

    def repr(self, x):
        return self.repr1(x, self.maxlevel)

    def repr1(self, x, level):
        typename = type(x).__name__
        if ' ' in typename:
            parts = typename.split()
            typename = '_'.join(parts)
        if hasattr(self, 'repr_' + typename):
            return getattr(self, 'repr_' + typename)(x, level)
        else:
            return self.repr_instance(x, level)

    def _repr_iterable(self, x, level, left, right, maxiter, trail=''):
        n = len(x)
        if level <= 0 and n:
            s = self.fillvalue
        else:
            newlevel = level - 1
            repr1 = self.repr1
            pieces = [repr1(elem, newlevel) for elem in islice(x, maxiter)]
            if n > maxiter:
                pieces.append(self.fillvalue)
            s = ', '.join(pieces)
            if n == 1 and trail:
                right = trail + right
        return '%s%s%s' % (left, s, right)

    def repr_tuple(self, x, level):
        return self._repr_iterable(x, level, '(', ')', self.maxtuple, ',')

    def repr_list(self, x, level):
        return self._repr_iterable(x, level, '[', ']', self.maxlist)

    def repr_array(self, x, level):
        if not x:
            return "array('%s')" % x.typecode
        header = "array('%s', [" % x.typecode
        return self._repr_iterable(x, level, header, '])', self.maxarray)

    def repr_set(self, x, level):
        if not x:
            return 'set()'
        x = _possibly_sorted(x)
        return self._repr_iterable(x, level, '{', '}', self.maxset)

    def repr_frozenset(self, x, level):
        if not x:
            return 'frozenset()'
        x = _possibly_sorted(x)
        return self._repr_iterable(x, level, 'frozenset({', '})',
                                   self.maxfrozenset)

    def repr_deque(self, x, level):
        return self._repr_iterable(x, level, 'deque([', '])', self.maxdeque)

    def repr_dict(self, x, level):
        n = len(x)
        if n == 0:
            return '{}'
        if level <= 0:
            return '{' + self.fillvalue + '}'
        newlevel = level - 1
        repr1 = self.repr1
        pieces = []
        for key in islice(_possibly_sorted(x), self.maxdict):
            keyrepr = repr1(key, newlevel)
            valrepr = repr1(x[key], newlevel)
            pieces.append('%s: %s' % (keyrepr, valrepr))
        if n > self.maxdict:
            pieces.append(self.fillvalue)
        s = ', '.join(pieces)
        return '{%s}' % (s,)

    def repr_str(self, x, level):
        s = builtins.repr(x[:self.maxstring])
        if len(s) > self.maxstring:
            i = max(0, (self.maxstring-3)//2)
            j = max(0, self.maxstring-3-i)
            s = builtins.repr(x[:i] + x[len(x)-j:])
            s = s[:i] + self.fillvalue + s[len(s)-j:]
        return s

    def repr_int(self, x, level):
        s = builtins.repr(x) # XXX Hope this isn't too slow...
        if len(s) > self.maxlong:
            i = max(0, (self.maxlong-3)//2)
            j = max(0, self.maxlong-3-i)
            s = s[:i] + self.fillvalue + s[len(s)-j:]
        return s

    def repr_instance(self, x, level):
        try:
            s = builtins.repr(x)
            # Bugs in x.__repr__() can cause arbitrary
            # exceptions -- then make up something
        except Exception:
            return '<%s instance at %#x>' % (x.__class__.__name__, id(x))
        if len(s) > self.maxother:
            i = max(0, (self.maxother-3)//2)
            j = max(0, self.maxother-3-i)
            s = s[:i] + self.fillvalue + s[len(s)-j:]
        return s


def _possibly_sorted(x):
    # Since not all sequences of items can be sorted and comparison
    # functions may raise arbitrary exceptions, return an unsorted
    # sequence in that case.
    try:
        return sorted(x)
    except Exception:
        return list(x)

aRepr = Repr()
repr = aRepr.repr
