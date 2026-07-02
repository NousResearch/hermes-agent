/* Set object interface */

#ifndef Py_SETOBJECT_H
#define Py_SETOBJECT_H
#ifdef __cplusplus
extern "C" {
#endif

PyAPI_DATA(PyTypeObject) PySet_Type;
PyAPI_DATA(PyTypeObject) PyFrozenSet_Type;
PyAPI_DATA(PyTypeObject) PySetIter_Type;

PyAPI_FUNC(PyObject *) PySet_New(PyObject *);
PyAPI_FUNC(PyObject *) PyFrozenSet_New(PyObject *);

PyAPI_FUNC(int) PySet_Add(PyObject *set, PyObject *key);
PyAPI_FUNC(int) PySet_Clear(PyObject *set);
PyAPI_FUNC(int) PySet_Contains(PyObject *anyset, PyObject *key);
PyAPI_FUNC(int) PySet_Discard(PyObject *set, PyObject *key);
PyAPI_FUNC(PyObject *) PySet_Pop(PyObject *set);
PyAPI_FUNC(Py_ssize_t) PySet_Size(PyObject *anyset);

#define PyFrozenSet_CheckExact(ob) Py_IS_TYPE(ob, &PyFrozenSet_Type)
#define PyFrozenSet_Check(ob) \
    (Py_IS_TYPE(ob, &PyFrozenSet_Type) || \
      PyType_IsSubtype(Py_TYPE(ob), &PyFrozenSet_Type))

#define PyAnySet_CheckExact(ob) \
    (Py_IS_TYPE(ob, &PySet_Type) || Py_IS_TYPE(ob, &PyFrozenSet_Type))
#define PyAnySet_Check(ob) \
    (Py_IS_TYPE(ob, &PySet_Type) || Py_IS_TYPE(ob, &PyFrozenSet_Type) || \
      PyType_IsSubtype(Py_TYPE(ob), &PySet_Type) || \
      PyType_IsSubtype(Py_TYPE(ob), &PyFrozenSet_Type))

#define PySet_CheckExact(op) Py_IS_TYPE(op, &PySet_Type)
#define PySet_Check(ob) \
    (Py_IS_TYPE(ob, &PySet_Type) || \
    PyType_IsSubtype(Py_TYPE(ob), &PySet_Type))

#ifndef Py_LIMITED_API
#  define Py_CPYTHON_SETOBJECT_H
#  include "cpython/setobject.h"
#  undef Py_CPYTHON_SETOBJECT_H
#endif

#ifdef __cplusplus
}
#endif
#endif /* !Py_SETOBJECT_H */
