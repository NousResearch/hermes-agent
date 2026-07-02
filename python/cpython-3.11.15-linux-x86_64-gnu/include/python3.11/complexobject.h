/* Complex number structure */

#ifndef Py_COMPLEXOBJECT_H
#define Py_COMPLEXOBJECT_H
#ifdef __cplusplus
extern "C" {
#endif

/* Complex object interface */

PyAPI_DATA(PyTypeObject) PyComplex_Type;

#define PyComplex_Check(op) PyObject_TypeCheck(op, &PyComplex_Type)
#define PyComplex_CheckExact(op) Py_IS_TYPE(op, &PyComplex_Type)

PyAPI_FUNC(PyObject *) PyComplex_FromDoubles(double real, double imag);

PyAPI_FUNC(double) PyComplex_RealAsDouble(PyObject *op);
PyAPI_FUNC(double) PyComplex_ImagAsDouble(PyObject *op);

#ifndef Py_LIMITED_API
#  define Py_CPYTHON_COMPLEXOBJECT_H
#  include "cpython/complexobject.h"
#  undef Py_CPYTHON_COMPLEXOBJECT_H
#endif

#ifdef __cplusplus
}
#endif
#endif /* !Py_COMPLEXOBJECT_H */
