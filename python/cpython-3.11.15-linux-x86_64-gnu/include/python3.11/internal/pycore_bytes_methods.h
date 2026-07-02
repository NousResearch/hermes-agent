#ifndef Py_LIMITED_API
#ifndef Py_BYTES_CTYPE_H
#define Py_BYTES_CTYPE_H

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

/*
 * The internal implementation behind PyBytes (bytes) and PyByteArray (bytearray)
 * methods of the given names, they operate on ASCII byte strings.
 */
extern PyObject* _Py_bytes_isspace(const char *cptr, Py_ssize_t len);
extern PyObject* _Py_bytes_isalpha(const char *cptr, Py_ssize_t len);
extern PyObject* _Py_bytes_isalnum(const char *cptr, Py_ssize_t len);
extern PyObject* _Py_bytes_isascii(const char *cptr, Py_ssize_t len);
extern PyObject* _Py_bytes_isdigit(const char *cptr, Py_ssize_t len);
extern PyObject* _Py_bytes_islower(const char *cptr, Py_ssize_t len);
extern PyObject* _Py_bytes_isupper(const char *cptr, Py_ssize_t len);
extern PyObject* _Py_bytes_istitle(const char *cptr, Py_ssize_t len);

/* These store their len sized answer in the given preallocated *result arg. */
extern void _Py_bytes_lower(char *result, const char *cptr, Py_ssize_t len);
extern void _Py_bytes_upper(char *result, const char *cptr, Py_ssize_t len);
extern void _Py_bytes_title(char *result, const char *s, Py_ssize_t len);
extern void _Py_bytes_capitalize(char *result, const char *s, Py_ssize_t len);
extern void _Py_bytes_swapcase(char *result, const char *s, Py_ssize_t len);

extern PyObject *_Py_bytes_find(const char *str, Py_ssize_t len, PyObject *args);
extern PyObject *_Py_bytes_index(const char *str, Py_ssize_t len, PyObject *args);
extern PyObject *_Py_bytes_rfind(const char *str, Py_ssize_t len, PyObject *args);
extern PyObject *_Py_bytes_rindex(const char *str, Py_ssize_t len, PyObject *args);
extern PyObject *_Py_bytes_count(const char *str, Py_ssize_t len, PyObject *args);
extern int _Py_bytes_contains(const char *str, Py_ssize_t len, PyObject *arg);
extern PyObject *_Py_bytes_startswith(const char *str, Py_ssize_t len, PyObject *args);
extern PyObject *_Py_bytes_endswith(const char *str, Py_ssize_t len, PyObject *args);

/* The maketrans() static method. */
extern PyObject* _Py_bytes_maketrans(Py_buffer *frm, Py_buffer *to);

/* Shared __doc__ strings. */
extern const char _Py_isspace__doc__[];
extern const char _Py_isalpha__doc__[];
extern const char _Py_isalnum__doc__[];
extern const char _Py_isascii__doc__[];
extern const char _Py_isdigit__doc__[];
extern const char _Py_islower__doc__[];
extern const char _Py_isupper__doc__[];
extern const char _Py_istitle__doc__[];
extern const char _Py_lower__doc__[];
extern const char _Py_upper__doc__[];
extern const char _Py_title__doc__[];
extern const char _Py_capitalize__doc__[];
extern const char _Py_swapcase__doc__[];
extern const char _Py_count__doc__[];
extern const char _Py_find__doc__[];
extern const char _Py_index__doc__[];
extern const char _Py_rfind__doc__[];
extern const char _Py_rindex__doc__[];
extern const char _Py_startswith__doc__[];
extern const char _Py_endswith__doc__[];
extern const char _Py_maketrans__doc__[];
extern const char _Py_expandtabs__doc__[];
extern const char _Py_ljust__doc__[];
extern const char _Py_rjust__doc__[];
extern const char _Py_center__doc__[];
extern const char _Py_zfill__doc__[];

/* this is needed because some docs are shared from the .o, not static */
#define PyDoc_STRVAR_shared(name,str) const char name[] = PyDoc_STR(str)

#endif /* !Py_BYTES_CTYPE_H */
#endif /* !Py_LIMITED_API */
