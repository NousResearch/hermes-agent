#ifndef Py_CPYTHON_COMPLEXOBJECT_H
#  error "this header file must not be included directly"
#endif

typedef struct {
    double real;
    double imag;
} Py_complex;

/* Operations on complex numbers from complexmodule.c */

PyAPI_FUNC(Py_complex) _Py_c_sum(Py_complex, Py_complex);
PyAPI_FUNC(Py_complex) _Py_c_diff(Py_complex, Py_complex);
PyAPI_FUNC(Py_complex) _Py_c_neg(Py_complex);
PyAPI_FUNC(Py_complex) _Py_c_prod(Py_complex, Py_complex);
PyAPI_FUNC(Py_complex) _Py_c_quot(Py_complex, Py_complex);
PyAPI_FUNC(Py_complex) _Py_c_pow(Py_complex, Py_complex);
PyAPI_FUNC(double) _Py_c_abs(Py_complex);

/* Complex object interface */

/*
PyComplexObject represents a complex number with double-precision
real and imaginary parts.
*/
typedef struct {
    PyObject_HEAD
    Py_complex cval;
} PyComplexObject;

PyAPI_FUNC(PyObject *) PyComplex_FromCComplex(Py_complex);

PyAPI_FUNC(Py_complex) PyComplex_AsCComplex(PyObject *op);

#ifdef Py_BUILD_CORE
/* Format the object based on the format_spec, as defined in PEP 3101
   (Advanced String Formatting). */
extern int _PyComplex_FormatAdvancedWriter(
    _PyUnicodeWriter *writer,
    PyObject *obj,
    PyObject *format_spec,
    Py_ssize_t start,
    Py_ssize_t end);
#endif  // Py_BUILD_CORE
