
/* Float object interface */

/*
PyFloatObject represents a (double precision) floating point number.
*/

#ifndef Py_FLOATOBJECT_H
#define Py_FLOATOBJECT_H
#ifdef __cplusplus
extern "C" {
#endif

PyAPI_DATA(PyTypeObject) PyFloat_Type;

#define PyFloat_Check(op) PyObject_TypeCheck(op, &PyFloat_Type)
#define PyFloat_CheckExact(op) Py_IS_TYPE(op, &PyFloat_Type)

#define Py_RETURN_NAN return PyFloat_FromDouble(Py_NAN)

#define Py_RETURN_INF(sign)                          \
    do {                                             \
        if (copysign(1., sign) == 1.) {              \
            return PyFloat_FromDouble(Py_HUGE_VAL);  \
        }                                            \
        else {                                       \
            return PyFloat_FromDouble(-Py_HUGE_VAL); \
        }                                            \
    } while(0)

PyAPI_FUNC(double) PyFloat_GetMax(void);
PyAPI_FUNC(double) PyFloat_GetMin(void);
PyAPI_FUNC(PyObject*) PyFloat_GetInfo(void);

/* Return Python float from string PyObject. */
PyAPI_FUNC(PyObject*) PyFloat_FromString(PyObject*);

/* Return Python float from C double. */
PyAPI_FUNC(PyObject*) PyFloat_FromDouble(double);

/* Extract C double from Python float.  The macro version trades safety for
   speed. */
PyAPI_FUNC(double) PyFloat_AsDouble(PyObject*);

#ifndef Py_LIMITED_API
#  define Py_CPYTHON_FLOATOBJECT_H
#  include "cpython/floatobject.h"
#  undef Py_CPYTHON_FLOATOBJECT_H
#endif

#ifdef __cplusplus
}
#endif
#endif /* !Py_FLOATOBJECT_H */
