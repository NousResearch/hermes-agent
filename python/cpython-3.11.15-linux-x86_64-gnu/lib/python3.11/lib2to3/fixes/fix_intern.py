# Copyright 2006 Georg Brandl.
# Licensed to PSF under a Contributor Agreement.

"""Fixer for intern().

intern(s) -> sys.intern(s)"""

# Local imports
from .. import fixer_base
from ..fixer_util import ImportAndCall, touch_import


class FixIntern(fixer_base.BaseFix):
    BM_compatible = True
    order = "pre"

    PATTERN = """
    power< 'intern'
           trailer< lpar='('
                    ( not(arglist | argument<any '=' any>) obj=any
                      | obj=arglist<(not argument<any '=' any>) any ','> )
                    rpar=')' >
           after=any*
    >
    """

    def transform(self, node, results):
        if results:
            # I feel like we should be able to express this logic in the
            # PATTERN above but I don't know how to do it so...
            obj = results['obj']
            if obj:
                if (obj.type == self.syms.argument and
                    obj.children[0].value in {'**', '*'}):
                    return  # Make no change.
        names = ('sys', 'intern')
        new = ImportAndCall(node, results, names)
        touch_import(None, 'sys', node)
        return new
