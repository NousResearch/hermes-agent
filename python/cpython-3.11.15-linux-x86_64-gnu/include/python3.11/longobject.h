#ifndef Py_LONGOBJECT_H
#define Py_LONGOBJECT_H
#ifdef __cplusplus
extern "C" {
#endif


/* Long (arbitrary precision) integer object interface */

PyAPI_DATA(PyTypeObject) PyLong_Type;

#define PyLong_Check(op) \
        PyType_FastSubclass(Py_TYPE(op), Py_TPFLAGS_LONG_SUBCLASS)
#define PyLong_CheckExact(op) Py_IS_TYPE(op, &PyLong_Type)

PyAPI_FUNC(PyObject *) PyLong_FromLong(long);
PyAPI_FUNC(PyObject *) PyLong_FromUnsignedLong(unsigned long);
PyAPI_FUNC(PyObject *) PyLong_FromSize_t(size_t);
PyAPI_FUNC(PyObject *) PyLong_FromSsize_t(Py_ssize_t);
PyAPI_FUNC(PyObject *) PyLong_FromDouble(double);
PyAPI_FUNC(long) PyLong_AsLong(PyObject *);
PyAPI_FUNC(long) PyLong_AsLongAndOverflow(PyObject *, int *);
PyAPI_FUNC(Py_ssize_t) PyLong_AsSsize_t(PyObject *);
PyAPI_FUNC(size_t) PyLong_AsSize_t(PyObject *);
PyAPI_FUNC(unsigned long) PyLong_AsUnsignedLong(PyObject *);
PyAPI_FUNC(unsigned long) PyLong_AsUnsignedLongMask(PyObject *);
PyAPI_FUNC(PyObject *) PyLong_GetInfo(void);

/* It may be useful in the future. I've added it in the PyInt -> PyLong
   cleanup to keep the extra information. [CH] */
#define PyLong_AS_LONG(op) PyLong_AsLong(op)

/* Issue #1983: pid_t can be longer than a C long on some systems */
#if !defined(SIZEOF_PID_T) || SIZEOF_PID_T == SIZEOF_INT
#define _Py_PARSE_PID "i"
#define PyLong_FromPid PyLong_FromLong
# ifndef Py_LIMITED_API
#   define PyLong_AsPid _PyLong_AsInt
# elif SIZEOF_INT == SIZEOF_LONG
#   define PyLong_AsPid PyLong_AsLong
# else
static inline int
PyLong_AsPid(PyObject *obj)
{
    int overflow;
    long result = PyLong_AsLongAndOverflow(obj, &overflow);
    if (overflow || result > INT_MAX || result < INT_MIN) {
        PyErr_SetString(PyExc_OverflowError,
                        "Python int too large to convert to C int");
        return -1;
    }
    return (int)result;
}
# endif
#elif SIZEOF_PID_T == SIZEOF_LONG
#define _Py_PARSE_PID "l"
#define PyLong_FromPid PyLong_FromLong
#define PyLong_AsPid PyLong_AsLong
#elif defined(SIZEOF_LONG_LONG) && SIZEOF_PID_T == SIZEOF_LONG_LONG
#define _Py_PARSE_PID "L"
#define PyLong_FromPid PyLong_FromLongLong
#define PyLong_AsPid PyLong_AsLongLong
#else
#error "sizeof(pid_t) is neither sizeof(int), sizeof(long) or sizeof(long long)"
#endif /* SIZEOF_PID_T */

#if SIZEOF_VOID_P == SIZEOF_INT
#  define _Py_PARSE_INTPTR "i"
#  define _Py_PARSE_UINTPTR "I"
#elif SIZEOF_VOID_P == SIZEOF_LONG
#  define _Py_PARSE_INTPTR "l"
#  define _Py_PARSE_UINTPTR "k"
#elif defined(SIZEOF_LONG_LONG) && SIZEOF_VOID_P == SIZEOF_LONG_LONG
#  define _Py_PARSE_INTPTR "L"
#  define _Py_PARSE_UINTPTR "K"
#else
#  error "void* different in size from int, long and long long"
#endif /* SIZEOF_VOID_P */

PyAPI_FUNC(double) PyLong_AsDouble(PyObject *);
PyAPI_FUNC(PyObject *) PyLong_FromVoidPtr(void *);
PyAPI_FUNC(void *) PyLong_AsVoidPtr(PyObject *);

PyAPI_FUNC(PyObject *) PyLong_FromLongLong(long long);
PyAPI_FUNC(PyObject *) PyLong_FromUnsignedLongLong(unsigned long long);
PyAPI_FUNC(long long) PyLong_AsLongLong(PyObject *);
PyAPI_FUNC(unsigned long long) PyLong_AsUnsignedLongLong(PyObject *);
PyAPI_FUNC(unsigned long long) PyLong_AsUnsignedLongLongMask(PyObject *);
PyAPI_FUNC(long long) PyLong_AsLongLongAndOverflow(PyObject *, int *);

PyAPI_FUNC(PyObject *) PyLong_FromString(const char *, char **, int);

/* These aren't really part of the int object, but they're handy. The
   functions are in Python/mystrtoul.c.
 */
PyAPI_FUNC(unsigned long) PyOS_strtoul(const char *, char **, int);
PyAPI_FUNC(long) PyOS_strtol(const char *, char **, int);

#ifndef Py_LIMITED_API
#  define Py_CPYTHON_LONGOBJECT_H
#  include "cpython/longobject.h"
#  undef Py_CPYTHON_LONGOBJECT_H
#endif

#ifdef __cplusplus
}
#endif
#endif /* !Py_LONGOBJECT_H */
