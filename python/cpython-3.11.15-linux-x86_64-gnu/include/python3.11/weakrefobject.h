/* Weak references objects for Python. */

#ifndef Py_WEAKREFOBJECT_H
#define Py_WEAKREFOBJECT_H
#ifdef __cplusplus
extern "C" {
#endif

typedef struct _PyWeakReference PyWeakReference;

PyAPI_DATA(PyTypeObject) _PyWeakref_RefType;
PyAPI_DATA(PyTypeObject) _PyWeakref_ProxyType;
PyAPI_DATA(PyTypeObject) _PyWeakref_CallableProxyType;

#define PyWeakref_CheckRef(op) PyObject_TypeCheck(op, &_PyWeakref_RefType)
#define PyWeakref_CheckRefExact(op) \
        Py_IS_TYPE(op, &_PyWeakref_RefType)
#define PyWeakref_CheckProxy(op) \
        (Py_IS_TYPE(op, &_PyWeakref_ProxyType) || \
         Py_IS_TYPE(op, &_PyWeakref_CallableProxyType))

#define PyWeakref_Check(op) \
        (PyWeakref_CheckRef(op) || PyWeakref_CheckProxy(op))


PyAPI_FUNC(PyObject *) PyWeakref_NewRef(PyObject *ob,
                                        PyObject *callback);
PyAPI_FUNC(PyObject *) PyWeakref_NewProxy(PyObject *ob,
                                          PyObject *callback);
PyAPI_FUNC(PyObject *) PyWeakref_GetObject(PyObject *ref);


#ifndef Py_LIMITED_API
#  define Py_CPYTHON_WEAKREFOBJECT_H
#  include "cpython/weakrefobject.h"
#  undef Py_CPYTHON_WEAKREFOBJECT_H
#endif

#ifdef __cplusplus
}
#endif
#endif /* !Py_WEAKREFOBJECT_H */
