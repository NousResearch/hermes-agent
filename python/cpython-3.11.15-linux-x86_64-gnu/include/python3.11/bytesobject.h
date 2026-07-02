
/* Bytes object interface */

#ifndef Py_BYTESOBJECT_H
#define Py_BYTESOBJECT_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>               // va_list

/*
Type PyBytesObject represents a byte string.  An extra zero byte is
reserved at the end to ensure it is zero-terminated, but a size is
present so strings with null bytes in them can be represented.  This
is an immutable object type.

There are functions to create new bytes objects, to test
an object for bytes-ness, and to get the
byte string value.  The latter function returns a null pointer
if the object is not of the proper type.
There is a variant that takes an explicit size as well as a
variant that assumes a zero-terminated string.  Note that none of the
functions should be applied to NULL pointer.
*/

PyAPI_DATA(PyTypeObject) PyBytes_Type;
PyAPI_DATA(PyTypeObject) PyBytesIter_Type;

#define PyBytes_Check(op) \
                 PyType_FastSubclass(Py_TYPE(op), Py_TPFLAGS_BYTES_SUBCLASS)
#define PyBytes_CheckExact(op) Py_IS_TYPE(op, &PyBytes_Type)

PyAPI_FUNC(PyObject *) PyBytes_FromStringAndSize(const char *, Py_ssize_t);
PyAPI_FUNC(PyObject *) PyBytes_FromString(const char *);
PyAPI_FUNC(PyObject *) PyBytes_FromObject(PyObject *);
PyAPI_FUNC(PyObject *) PyBytes_FromFormatV(const char*, va_list)
                                Py_GCC_ATTRIBUTE((format(printf, 1, 0)));
PyAPI_FUNC(PyObject *) PyBytes_FromFormat(const char*, ...)
                                Py_GCC_ATTRIBUTE((format(printf, 1, 2)));
PyAPI_FUNC(Py_ssize_t) PyBytes_Size(PyObject *);
PyAPI_FUNC(char *) PyBytes_AsString(PyObject *);
PyAPI_FUNC(PyObject *) PyBytes_Repr(PyObject *, int);
PyAPI_FUNC(void) PyBytes_Concat(PyObject **, PyObject *);
PyAPI_FUNC(void) PyBytes_ConcatAndDel(PyObject **, PyObject *);
PyAPI_FUNC(PyObject *) PyBytes_DecodeEscape(const char *, Py_ssize_t,
                                            const char *, Py_ssize_t,
                                            const char *);

/* Provides access to the internal data buffer and size of a bytes object.
   Passing NULL as len parameter will force the string buffer to be
   0-terminated (passing a string with embedded NUL characters will
   cause an exception).  */
PyAPI_FUNC(int) PyBytes_AsStringAndSize(
    PyObject *obj,      /* bytes object */
    char **s,           /* pointer to buffer variable */
    Py_ssize_t *len     /* pointer to length variable or NULL */
    );

#ifndef Py_LIMITED_API
#  define Py_CPYTHON_BYTESOBJECT_H
#  include "cpython/bytesobject.h"
#  undef Py_CPYTHON_BYTESOBJECT_H
#endif

#ifdef __cplusplus
}
#endif
#endif /* !Py_BYTESOBJECT_H */
