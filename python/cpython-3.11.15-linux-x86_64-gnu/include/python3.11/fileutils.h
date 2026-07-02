#ifndef Py_FILEUTILS_H
#define Py_FILEUTILS_H
#ifdef __cplusplus
extern "C" {
#endif

#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x03050000
PyAPI_FUNC(wchar_t *) Py_DecodeLocale(
    const char *arg,
    size_t *size);

PyAPI_FUNC(char*) Py_EncodeLocale(
    const wchar_t *text,
    size_t *error_pos);
#endif

#ifndef Py_LIMITED_API
#  define Py_CPYTHON_FILEUTILS_H
#  include "cpython/fileutils.h"
#  undef Py_CPYTHON_FILEUTILS_H
#endif

#ifdef __cplusplus
}
#endif
#endif /* !Py_FILEUTILS_H */
