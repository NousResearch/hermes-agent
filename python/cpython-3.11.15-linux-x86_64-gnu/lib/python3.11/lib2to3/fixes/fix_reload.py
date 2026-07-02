"""Fixer for reload().

reload(s) -> importlib.reload(s)"""

# Local imports
from .. import fixer_base
from ..fixer_util import ImportAndCall, touch_import


class FixReload(fixer_base.BaseFix):
    BM_compatible = True
    order = "pre"

    PATTERN = """
    power< 'reload'
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
        names = ('importlib', 'reload')
        new = ImportAndCall(node, results, names)
        touch_import(None, 'importlib', node)
        return new
