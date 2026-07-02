/* ByteArray object interface */

#ifndef Py_BYTEARRAYOBJECT_H
#define Py_BYTEARRAYOBJECT_H
#ifdef __cplusplus
extern "C" {
#endif

/* Type PyByteArrayObject represents a mutable array of bytes.
 * The Python API is that of a sequence;
 * the bytes are mapped to ints in [0, 256).
 * Bytes are not characters; they may be used to encode characters.
 * The only way to go between bytes and str/unicode is via encoding
 * and decoding.
 * For the convenience of C programmers, the bytes type is considered
 * to contain a char pointer, not an unsigned char pointer.
 */

/* Type object */
PyAPI_DATA(PyTypeObject) PyByteArray_Type;
PyAPI_DATA(PyTypeObject) PyByteArrayIter_Type;

/* Type check macros */
#define PyByteArray_Check(self) PyObject_TypeCheck(self, &PyByteArray_Type)
#define PyByteArray_CheckExact(self) Py_IS_TYPE(self, &PyByteArray_Type)

/* Direct API functions */
PyAPI_FUNC(PyObject *) PyByteArray_FromObject(PyObject *);
PyAPI_FUNC(PyObject *) PyByteArray_Concat(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyByteArray_FromStringAndSize(const char *, Py_ssize_t);
PyAPI_FUNC(Py_ssize_t) PyByteArray_Size(PyObject *);
PyAPI_FUNC(char *) PyByteArray_AsString(PyObject *);
PyAPI_FUNC(int) PyByteArray_Resize(PyObject *, Py_ssize_t);

#ifndef Py_LIMITED_API
#  define Py_CPYTHON_BYTEARRAYOBJECT_H
#  include "cpython/bytearrayobject.h"
#  undef Py_CPYTHON_BYTEARRAYOBJECT_H
#endif

#ifdef __cplusplus
}
#endif
#endif /* !Py_BYTEARRAYOBJECT_H */
