// Simple namespace object interface

#ifndef Py_INTERNAL_NAMESPACE_H
#define Py_INTERNAL_NAMESPACE_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

PyAPI_DATA(PyTypeObject) _PyNamespace_Type;

PyAPI_FUNC(PyObject *) _PyNamespace_New(PyObject *kwds);

#ifdef __cplusplus
}
#endif
#endif  // !Py_INTERNAL_NAMESPACE_H
