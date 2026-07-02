/* List object interface

   Another generally useful object type is a list of object pointers.
   This is a mutable type: the list items can be changed, and items can be
   added or removed. Out-of-range indices or non-list objects are ignored.

   WARNING: PyList_SetItem does not increment the new item's reference count,
   but does decrement the reference count of the item it replaces, if not nil.
   It does *decrement* the reference count if it is *not* inserted in the list.
   Similarly, PyList_GetItem does not increment the returned item's reference
   count.
*/

#ifndef Py_LISTOBJECT_H
#define Py_LISTOBJECT_H
#ifdef __cplusplus
extern "C" {
#endif

PyAPI_DATA(PyTypeObject) PyList_Type;
PyAPI_DATA(PyTypeObject) PyListIter_Type;
PyAPI_DATA(PyTypeObject) PyListRevIter_Type;

#define PyList_Check(op) \
    PyType_FastSubclass(Py_TYPE(op), Py_TPFLAGS_LIST_SUBCLASS)
#define PyList_CheckExact(op) Py_IS_TYPE(op, &PyList_Type)

PyAPI_FUNC(PyObject *) PyList_New(Py_ssize_t size);
PyAPI_FUNC(Py_ssize_t) PyList_Size(PyObject *);

PyAPI_FUNC(PyObject *) PyList_GetItem(PyObject *, Py_ssize_t);
PyAPI_FUNC(int) PyList_SetItem(PyObject *, Py_ssize_t, PyObject *);
PyAPI_FUNC(int) PyList_Insert(PyObject *, Py_ssize_t, PyObject *);
PyAPI_FUNC(int) PyList_Append(PyObject *, PyObject *);

PyAPI_FUNC(PyObject *) PyList_GetSlice(PyObject *, Py_ssize_t, Py_ssize_t);
PyAPI_FUNC(int) PyList_SetSlice(PyObject *, Py_ssize_t, Py_ssize_t, PyObject *);

PyAPI_FUNC(int) PyList_Sort(PyObject *);
PyAPI_FUNC(int) PyList_Reverse(PyObject *);
PyAPI_FUNC(PyObject *) PyList_AsTuple(PyObject *);

#ifndef Py_LIMITED_API
#  define Py_CPYTHON_LISTOBJECT_H
#  include "cpython/listobject.h"
#  undef Py_CPYTHON_LISTOBJECT_H
#endif

#ifdef __cplusplus
}
#endif
#endif /* !Py_LISTOBJECT_H */
