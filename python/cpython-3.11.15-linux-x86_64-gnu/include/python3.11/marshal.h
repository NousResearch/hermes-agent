
/* Interface for marshal.c */

#ifndef Py_MARSHAL_H
#define Py_MARSHAL_H
#ifndef Py_LIMITED_API

#ifdef __cplusplus
extern "C" {
#endif

PyAPI_FUNC(PyObject *) PyMarshal_ReadObjectFromString(const char *,
                                                      Py_ssize_t);
PyAPI_FUNC(PyObject *) PyMarshal_WriteObjectToString(PyObject *, int);

#define Py_MARSHAL_VERSION 4

PyAPI_FUNC(long) PyMarshal_ReadLongFromFile(FILE *);
PyAPI_FUNC(int) PyMarshal_ReadShortFromFile(FILE *);
PyAPI_FUNC(PyObject *) PyMarshal_ReadObjectFromFile(FILE *);
PyAPI_FUNC(PyObject *) PyMarshal_ReadLastObjectFromFile(FILE *);

PyAPI_FUNC(void) PyMarshal_WriteLongToFile(long, FILE *, int);
PyAPI_FUNC(void) PyMarshal_WriteObjectToFile(PyObject *, FILE *, int);

#ifdef __cplusplus
}
#endif

#endif /* Py_LIMITED_API */
#endif /* !Py_MARSHAL_H */
