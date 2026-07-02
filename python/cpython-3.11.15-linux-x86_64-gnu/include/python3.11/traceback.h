#ifndef Py_TRACEBACK_H
#define Py_TRACEBACK_H
#ifdef __cplusplus
extern "C" {
#endif

/* Traceback interface */

PyAPI_FUNC(int) PyTraceBack_Here(PyFrameObject *);
PyAPI_FUNC(int) PyTraceBack_Print(PyObject *, PyObject *);

/* Reveal traceback type so we can typecheck traceback objects */
PyAPI_DATA(PyTypeObject) PyTraceBack_Type;
#define PyTraceBack_Check(v) Py_IS_TYPE(v, &PyTraceBack_Type)


#ifndef Py_LIMITED_API
#  define Py_CPYTHON_TRACEBACK_H
#  include "cpython/traceback.h"
#  undef Py_CPYTHON_TRACEBACK_H
#endif

#ifdef __cplusplus
}
#endif
#endif /* !Py_TRACEBACK_H */
