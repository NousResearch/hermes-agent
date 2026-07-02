"""Fixer for sys.exc_{type, value, traceback}

sys.exc_type -> sys.exc_info()[0]
sys.exc_value -> sys.exc_info()[1]
sys.exc_traceback -> sys.exc_info()[2]
"""

# By Jeff Balogh and Benjamin Peterson

# Local imports
from .. import fixer_base
from ..fixer_util import Attr, Call, Name, Number, Subscript, Node, syms

class FixSysExc(fixer_base.BaseFix):
    # This order matches the ordering of sys.exc_info().
    exc_info = ["exc_type", "exc_value", "exc_traceback"]
    BM_compatible = True
    PATTERN = """
              power< 'sys' trailer< dot='.' attribute=(%s) > >
              """ % '|'.join("'%s'" % e for e in exc_info)

    def transform(self, node, results):
        sys_attr = results["attribute"][0]
        index = Number(self.exc_info.index(sys_attr.value))

        call = Call(Name("exc_info"), prefix=sys_attr.prefix)
        attr = Attr(Name("sys"), call)
        attr[1].children[0].prefix = results["dot"].prefix
        attr.append(Subscript(index))
        return Node(syms.power, attr, prefix=node.prefix)
