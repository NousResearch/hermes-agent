"""Fixer for operator functions.

operator.isCallable(obj)       -> callable(obj)
operator.sequenceIncludes(obj) -> operator.contains(obj)
operator.isSequenceType(obj)   -> isinstance(obj, collections.abc.Sequence)
operator.isMappingType(obj)    -> isinstance(obj, collections.abc.Mapping)
operator.isNumberType(obj)     -> isinstance(obj, numbers.Number)
operator.repeat(obj, n)        -> operator.mul(obj, n)
operator.irepeat(obj, n)       -> operator.imul(obj, n)
"""

import collections.abc

# Local imports
from lib2to3 import fixer_base
from lib2to3.fixer_util import Call, Name, String, touch_import


def invocation(s):
    def dec(f):
        f.invocation = s
        return f
    return dec


class FixOperator(fixer_base.BaseFix):
    BM_compatible = True
    order = "pre"

    methods = """
              method=('isCallable'|'sequenceIncludes'
                     |'isSequenceType'|'isMappingType'|'isNumberType'
                     |'repeat'|'irepeat')
              """
    obj = "'(' obj=any ')'"
    PATTERN = """
              power< module='operator'
                trailer< '.' %(methods)s > trailer< %(obj)s > >
              |
              power< %(methods)s trailer< %(obj)s > >
              """ % dict(methods=methods, obj=obj)

    def transform(self, node, results):
        method = self._check_method(node, results)
        if method is not None:
            return method(node, results)

    @invocation("operator.contains(%s)")
    def _sequenceIncludes(self, node, results):
        return self._handle_rename(node, results, "contains")

    @invocation("callable(%s)")
    def _isCallable(self, node, results):
        obj = results["obj"]
        return Call(Name("callable"), [obj.clone()], prefix=node.prefix)

    @invocation("operator.mul(%s)")
    def _repeat(self, node, results):
        return self._handle_rename(node, results, "mul")

    @invocation("operator.imul(%s)")
    def _irepeat(self, node, results):
        return self._handle_rename(node, results, "imul")

    @invocation("isinstance(%s, collections.abc.Sequence)")
    def _isSequenceType(self, node, results):
        return self._handle_type2abc(node, results, "collections.abc", "Sequence")

    @invocation("isinstance(%s, collections.abc.Mapping)")
    def _isMappingType(self, node, results):
        return self._handle_type2abc(node, results, "collections.abc", "Mapping")

    @invocation("isinstance(%s, numbers.Number)")
    def _isNumberType(self, node, results):
        return self._handle_type2abc(node, results, "numbers", "Number")

    def _handle_rename(self, node, results, name):
        method = results["method"][0]
        method.value = name
        method.changed()

    def _handle_type2abc(self, node, results, module, abc):
        touch_import(None, module, node)
        obj = results["obj"]
        args = [obj.clone(), String(", " + ".".join([module, abc]))]
        return Call(Name("isinstance"), args, prefix=node.prefix)

    def _check_method(self, node, results):
        method = getattr(self, "_" + results["method"][0].value)
        if isinstance(method, collections.abc.Callable):
            if "module" in results:
                return method
            else:
                sub = (str(results["obj"]),)
                invocation_str = method.invocation % sub
                self.warning(node, "You should use '%s' here." % invocation_str)
        return None
