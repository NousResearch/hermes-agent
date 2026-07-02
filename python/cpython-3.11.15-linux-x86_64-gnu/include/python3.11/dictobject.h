#ifndef Py_DICTOBJECT_H
#define Py_DICTOBJECT_H
#ifdef __cplusplus
extern "C" {
#endif

/* Dictionary object type -- mapping from hashable object to object */

/* The distribution includes a separate file, Objects/dictnotes.txt,
   describing explorations into dictionary design and optimization.
   It covers typical dictionary use patterns, the parameters for
   tuning dictionaries, and several ideas for possible optimizations.
*/

PyAPI_DATA(PyTypeObject) PyDict_Type;

#define PyDict_Check(op) \
                 PyType_FastSubclass(Py_TYPE(op), Py_TPFLAGS_DICT_SUBCLASS)
#define PyDict_CheckExact(op) Py_IS_TYPE(op, &PyDict_Type)

PyAPI_FUNC(PyObject *) PyDict_New(void);
PyAPI_FUNC(PyObject *) PyDict_GetItem(PyObject *mp, PyObject *key);
PyAPI_FUNC(PyObject *) PyDict_GetItemWithError(PyObject *mp, PyObject *key);
PyAPI_FUNC(int) PyDict_SetItem(PyObject *mp, PyObject *key, PyObject *item);
PyAPI_FUNC(int) PyDict_DelItem(PyObject *mp, PyObject *key);
PyAPI_FUNC(void) PyDict_Clear(PyObject *mp);
PyAPI_FUNC(int) PyDict_Next(
    PyObject *mp, Py_ssize_t *pos, PyObject **key, PyObject **value);
PyAPI_FUNC(PyObject *) PyDict_Keys(PyObject *mp);
PyAPI_FUNC(PyObject *) PyDict_Values(PyObject *mp);
PyAPI_FUNC(PyObject *) PyDict_Items(PyObject *mp);
PyAPI_FUNC(Py_ssize_t) PyDict_Size(PyObject *mp);
PyAPI_FUNC(PyObject *) PyDict_Copy(PyObject *mp);
PyAPI_FUNC(int) PyDict_Contains(PyObject *mp, PyObject *key);

/* PyDict_Update(mp, other) is equivalent to PyDict_Merge(mp, other, 1). */
PyAPI_FUNC(int) PyDict_Update(PyObject *mp, PyObject *other);

/* PyDict_Merge updates/merges from a mapping object (an object that
   supports PyMapping_Keys() and PyObject_GetItem()).  If override is true,
   the last occurrence of a key wins, else the first.  The Python
   dict.update(other) is equivalent to PyDict_Merge(dict, other, 1).
*/
PyAPI_FUNC(int) PyDict_Merge(PyObject *mp,
                             PyObject *other,
                             int override);

/* PyDict_MergeFromSeq2 updates/merges from an iterable object producing
   iterable objects of length 2.  If override is true, the last occurrence
   of a key wins, else the first.  The Python dict constructor dict(seq2)
   is equivalent to dict={}; PyDict_MergeFromSeq(dict, seq2, 1).
*/
PyAPI_FUNC(int) PyDict_MergeFromSeq2(PyObject *d,
                                     PyObject *seq2,
                                     int override);

PyAPI_FUNC(PyObject *) PyDict_GetItemString(PyObject *dp, const char *key);
PyAPI_FUNC(int) PyDict_SetItemString(PyObject *dp, const char *key, PyObject *item);
PyAPI_FUNC(int) PyDict_DelItemString(PyObject *dp, const char *key);
#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x030A0000
PyAPI_FUNC(PyObject *) PyObject_GenericGetDict(PyObject *, void *);
#endif

/* Dictionary (keys, values, items) views */

PyAPI_DATA(PyTypeObject) PyDictKeys_Type;
PyAPI_DATA(PyTypeObject) PyDictValues_Type;
PyAPI_DATA(PyTypeObject) PyDictItems_Type;

#define PyDictKeys_Check(op) PyObject_TypeCheck(op, &PyDictKeys_Type)
#define PyDictValues_Check(op) PyObject_TypeCheck(op, &PyDictValues_Type)
#define PyDictItems_Check(op) PyObject_TypeCheck(op, &PyDictItems_Type)
/* This excludes Values, since they are not sets. */
# define PyDictViewSet_Check(op) \
    (PyDictKeys_Check(op) || PyDictItems_Check(op))

/* Dictionary (key, value, items) iterators */

PyAPI_DATA(PyTypeObject) PyDictIterKey_Type;
PyAPI_DATA(PyTypeObject) PyDictIterValue_Type;
PyAPI_DATA(PyTypeObject) PyDictIterItem_Type;

PyAPI_DATA(PyTypeObject) PyDictRevIterKey_Type;
PyAPI_DATA(PyTypeObject) PyDictRevIterItem_Type;
PyAPI_DATA(PyTypeObject) PyDictRevIterValue_Type;


#ifndef Py_LIMITED_API
#  define Py_CPYTHON_DICTOBJECT_H
#  include "cpython/dictobject.h"
#  undef Py_CPYTHON_DICTOBJECT_H
#endif

#ifdef __cplusplus
}
#endif
#endif /* !Py_DICTOBJECT_H */
