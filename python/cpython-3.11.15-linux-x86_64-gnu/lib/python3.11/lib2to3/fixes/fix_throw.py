"""Fixer for generator.throw(E, V, T).

g.throw(E)       -> g.throw(E)
g.throw(E, V)    -> g.throw(E(V))
g.throw(E, V, T) -> g.throw(E(V).with_traceback(T))

g.throw("foo"[, V[, T]]) will warn about string exceptions."""
# Author: Collin Winter

# Local imports
from .. import pytree
from ..pgen2 import token
from .. import fixer_base
from ..fixer_util import Name, Call, ArgList, Attr, is_tuple

class FixThrow(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = """
    power< any trailer< '.' 'throw' >
           trailer< '(' args=arglist< exc=any ',' val=any [',' tb=any] > ')' >
    >
    |
    power< any trailer< '.' 'throw' > trailer< '(' exc=any ')' > >
    """

    def transform(self, node, results):
        syms = self.syms

        exc = results["exc"].clone()
        if exc.type is token.STRING:
            self.cannot_convert(node, "Python 3 does not support string exceptions")
            return

        # Leave "g.throw(E)" alone
        val = results.get("val")
        if val is None:
            return

        val = val.clone()
        if is_tuple(val):
            args = [c.clone() for c in val.children[1:-1]]
        else:
            val.prefix = ""
            args = [val]

        throw_args = results["args"]

        if "tb" in results:
            tb = results["tb"].clone()
            tb.prefix = ""

            e = Call(exc, args)
            with_tb = Attr(e, Name('with_traceback')) + [ArgList([tb])]
            throw_args.replace(pytree.Node(syms.power, with_tb))
        else:
            throw_args.replace(Call(exc, args))
