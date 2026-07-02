# Copyright 2007 Google, Inc. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

"""Fixer for removing uses of the types module.

These work for only the known names in the types module.  The forms above
can include types. or not.  ie, It is assumed the module is imported either as:

    import types
    from types import ... # either * or specific types

The import statements are not modified.

There should be another fixer that handles at least the following constants:

   type([]) -> list
   type(()) -> tuple
   type('') -> str

"""

# Local imports
from .. import fixer_base
from ..fixer_util import Name

_TYPE_MAPPING = {
        'BooleanType' : 'bool',
        'BufferType' : 'memoryview',
        'ClassType' : 'type',
        'ComplexType' : 'complex',
        'DictType': 'dict',
        'DictionaryType' : 'dict',
        'EllipsisType' : 'type(Ellipsis)',
        #'FileType' : 'io.IOBase',
        'FloatType': 'float',
        'IntType': 'int',
        'ListType': 'list',
        'LongType': 'int',
        'ObjectType' : 'object',
        'NoneType': 'type(None)',
        'NotImplementedType' : 'type(NotImplemented)',
        'SliceType' : 'slice',
        'StringType': 'bytes', # XXX ?
        'StringTypes' : '(str,)', # XXX ?
        'TupleType': 'tuple',
        'TypeType' : 'type',
        'UnicodeType': 'str',
        'XRangeType' : 'range',
    }

_pats = ["power< 'types' trailer< '.' name='%s' > >" % t for t in _TYPE_MAPPING]

class FixTypes(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = '|'.join(_pats)

    def transform(self, node, results):
        new_value = _TYPE_MAPPING.get(results["name"].value)
        if new_value:
            return Name(new_value, prefix=node.prefix)
        return None
