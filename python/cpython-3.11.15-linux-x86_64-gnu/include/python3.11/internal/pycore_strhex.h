#ifndef Py_INTERNAL_STRHEX_H
#define Py_INTERNAL_STRHEX_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

// Returns a str() containing the hex representation of argbuf.
PyAPI_FUNC(PyObject*) _Py_strhex(const
    char* argbuf,
    const Py_ssize_t arglen);

// Returns a bytes() containing the ASCII hex representation of argbuf.
PyAPI_FUNC(PyObject*) _Py_strhex_bytes(
    const char* argbuf,
    const Py_ssize_t arglen);

// These variants include support for a separator between every N bytes:
PyAPI_FUNC(PyObject*) _Py_strhex_with_sep(
    const char* argbuf,
    const Py_ssize_t arglen,
    PyObject* sep,
    const int bytes_per_group);
PyAPI_FUNC(PyObject*) _Py_strhex_bytes_with_sep(
    const char* argbuf,
    const Py_ssize_t arglen,
    PyObject* sep,
    const int bytes_per_group);

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_STRHEX_H */
