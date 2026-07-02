""" Fixer for itertools.(imap|ifilter|izip) --> (map|filter|zip) and
    itertools.ifilterfalse --> itertools.filterfalse (bugs 2360-2363)

    imports from itertools are fixed in fix_itertools_import.py

    If itertools is imported as something else (ie: import itertools as it;
    it.izip(spam, eggs)) method calls will not get fixed.
    """

# Local imports
from .. import fixer_base
from ..fixer_util import Name

class FixItertools(fixer_base.BaseFix):
    BM_compatible = True
    it_funcs = "('imap'|'ifilter'|'izip'|'izip_longest'|'ifilterfalse')"
    PATTERN = """
              power< it='itertools'
                  trailer<
                     dot='.' func=%(it_funcs)s > trailer< '(' [any] ')' > >
              |
              power< func=%(it_funcs)s trailer< '(' [any] ')' > >
              """ %(locals())

    # Needs to be run after fix_(map|zip|filter)
    run_order = 6

    def transform(self, node, results):
        prefix = None
        func = results['func'][0]
        if ('it' in results and
            func.value not in ('ifilterfalse', 'izip_longest')):
            dot, it = (results['dot'], results['it'])
            # Remove the 'itertools'
            prefix = it.prefix
            it.remove()
            # Replace the node which contains ('.', 'function') with the
            # function (to be consistent with the second part of the pattern)
            dot.remove()
            func.parent.replace(func)

        prefix = prefix or func.prefix
        func.replace(Name(func.value[1:], prefix=prefix))
