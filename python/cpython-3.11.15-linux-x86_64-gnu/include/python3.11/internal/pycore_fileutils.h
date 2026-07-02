#ifndef Py_INTERNAL_FILEUTILS_H
#define Py_INTERNAL_FILEUTILS_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "Py_BUILD_CORE must be defined to include this header"
#endif

#include <locale.h>   /* struct lconv */

typedef enum {
    _Py_ERROR_UNKNOWN=0,
    _Py_ERROR_STRICT,
    _Py_ERROR_SURROGATEESCAPE,
    _Py_ERROR_REPLACE,
    _Py_ERROR_IGNORE,
    _Py_ERROR_BACKSLASHREPLACE,
    _Py_ERROR_SURROGATEPASS,
    _Py_ERROR_XMLCHARREFREPLACE,
    _Py_ERROR_OTHER
} _Py_error_handler;

PyAPI_FUNC(_Py_error_handler) _Py_GetErrorHandler(const char *errors);

PyAPI_FUNC(int) _Py_DecodeLocaleEx(
    const char *arg,
    wchar_t **wstr,
    size_t *wlen,
    const char **reason,
    int current_locale,
    _Py_error_handler errors);

PyAPI_FUNC(int) _Py_EncodeLocaleEx(
    const wchar_t *text,
    char **str,
    size_t *error_pos,
    const char **reason,
    int current_locale,
    _Py_error_handler errors);

PyAPI_FUNC(char*) _Py_EncodeLocaleRaw(
    const wchar_t *text,
    size_t *error_pos);

PyAPI_FUNC(PyObject *) _Py_device_encoding(int);

#if defined(MS_WINDOWS) || defined(__APPLE__)
    /* On Windows, the count parameter of read() is an int (bpo-9015, bpo-9611).
       On macOS 10.13, read() and write() with more than INT_MAX bytes
       fail with EINVAL (bpo-24658). */
#   define _PY_READ_MAX  INT_MAX
#   define _PY_WRITE_MAX INT_MAX
#else
    /* write() should truncate the input to PY_SSIZE_T_MAX bytes,
       but it's safer to do it ourself to have a portable behaviour */
#   define _PY_READ_MAX  PY_SSIZE_T_MAX
#   define _PY_WRITE_MAX PY_SSIZE_T_MAX
#endif

#ifdef MS_WINDOWS
struct _Py_stat_struct {
    unsigned long st_dev;
    uint64_t st_ino;
    unsigned short st_mode;
    int st_nlink;
    int st_uid;
    int st_gid;
    unsigned long st_rdev;
    __int64 st_size;
    time_t st_atime;
    int st_atime_nsec;
    time_t st_mtime;
    int st_mtime_nsec;
    time_t st_ctime;
    int st_ctime_nsec;
    unsigned long st_file_attributes;
    unsigned long st_reparse_tag;
};
#else
#  define _Py_stat_struct stat
#endif

PyAPI_FUNC(int) _Py_fstat(
    int fd,
    struct _Py_stat_struct *status);

PyAPI_FUNC(int) _Py_fstat_noraise(
    int fd,
    struct _Py_stat_struct *status);

PyAPI_FUNC(int) _Py_stat(
    PyObject *path,
    struct stat *status);

PyAPI_FUNC(int) _Py_open(
    const char *pathname,
    int flags);

PyAPI_FUNC(int) _Py_open_noraise(
    const char *pathname,
    int flags);

PyAPI_FUNC(FILE *) _Py_wfopen(
    const wchar_t *path,
    const wchar_t *mode);

PyAPI_FUNC(Py_ssize_t) _Py_read(
    int fd,
    void *buf,
    size_t count);

PyAPI_FUNC(Py_ssize_t) _Py_write(
    int fd,
    const void *buf,
    size_t count);

PyAPI_FUNC(Py_ssize_t) _Py_write_noraise(
    int fd,
    const void *buf,
    size_t count);

#ifdef HAVE_READLINK
PyAPI_FUNC(int) _Py_wreadlink(
    const wchar_t *path,
    wchar_t *buf,
    /* Number of characters of 'buf' buffer
       including the trailing NUL character */
    size_t buflen);
#endif

#ifdef HAVE_REALPATH
PyAPI_FUNC(wchar_t*) _Py_wrealpath(
    const wchar_t *path,
    wchar_t *resolved_path,
    /* Number of characters of 'resolved_path' buffer
       including the trailing NUL character */
    size_t resolved_path_len);
#endif

PyAPI_FUNC(wchar_t*) _Py_wgetcwd(
    wchar_t *buf,
    /* Number of characters of 'buf' buffer
       including the trailing NUL character */
    size_t buflen);

PyAPI_FUNC(int) _Py_get_inheritable(int fd);

PyAPI_FUNC(int) _Py_set_inheritable(int fd, int inheritable,
                                    int *atomic_flag_works);

PyAPI_FUNC(int) _Py_set_inheritable_async_safe(int fd, int inheritable,
                                               int *atomic_flag_works);

PyAPI_FUNC(int) _Py_dup(int fd);

#ifndef MS_WINDOWS
PyAPI_FUNC(int) _Py_get_blocking(int fd);

PyAPI_FUNC(int) _Py_set_blocking(int fd, int blocking);
#else   /* MS_WINDOWS */
PyAPI_FUNC(void*) _Py_get_osfhandle_noraise(int fd);

PyAPI_FUNC(void*) _Py_get_osfhandle(int fd);

PyAPI_FUNC(int) _Py_open_osfhandle_noraise(void *handle, int flags);

PyAPI_FUNC(int) _Py_open_osfhandle(void *handle, int flags);
#endif  /* MS_WINDOWS */

// This is used after getting NULL back from Py_DecodeLocale().
#define DECODE_LOCALE_ERR(NAME, LEN) \
    ((LEN) == (size_t)-2) \
     ? _PyStatus_ERR("cannot decode " NAME) \
     : _PyStatus_NO_MEMORY()

PyAPI_DATA(int) _Py_HasFileSystemDefaultEncodeErrors;

PyAPI_FUNC(int) _Py_DecodeUTF8Ex(
    const char *arg,
    Py_ssize_t arglen,
    wchar_t **wstr,
    size_t *wlen,
    const char **reason,
    _Py_error_handler errors);

PyAPI_FUNC(int) _Py_EncodeUTF8Ex(
    const wchar_t *text,
    char **str,
    size_t *error_pos,
    const char **reason,
    int raw_malloc,
    _Py_error_handler errors);

PyAPI_FUNC(wchar_t*) _Py_DecodeUTF8_surrogateescape(
    const char *arg,
    Py_ssize_t arglen,
    size_t *wlen);

extern int
_Py_wstat(const wchar_t *, struct stat *);

PyAPI_FUNC(int) _Py_GetForceASCII(void);

/* Reset "force ASCII" mode (if it was initialized).

   This function should be called when Python changes the LC_CTYPE locale,
   so the "force ASCII" mode can be detected again on the new locale
   encoding. */
PyAPI_FUNC(void) _Py_ResetForceASCII(void);


PyAPI_FUNC(int) _Py_GetLocaleconvNumeric(
    struct lconv *lc,
    PyObject **decimal_point,
    PyObject **thousands_sep);

PyAPI_FUNC(void) _Py_closerange(int first, int last);

PyAPI_FUNC(wchar_t*) _Py_GetLocaleEncoding(void);
PyAPI_FUNC(PyObject*) _Py_GetLocaleEncodingObject(void);

#ifdef HAVE_NON_UNICODE_WCHAR_T_REPRESENTATION
extern int _Py_LocaleUsesNonUnicodeWchar(void);

extern wchar_t* _Py_DecodeNonUnicodeWchar(
    const wchar_t* native,
    Py_ssize_t size);

extern int _Py_EncodeNonUnicodeWchar_InPlace(
    wchar_t* unicode,
    Py_ssize_t size);
#endif

extern int _Py_isabs(const wchar_t *path);
extern int _Py_abspath(const wchar_t *path, wchar_t **abspath_p);
#ifdef MS_WINDOWS
extern int _PyOS_getfullpathname(const wchar_t *path, wchar_t **abspath_p);
#endif
extern wchar_t * _Py_join_relfile(const wchar_t *dirname,
                                  const wchar_t *relfile);
extern int _Py_add_relfile(wchar_t *dirname,
                           const wchar_t *relfile,
                           size_t bufsize);
extern size_t _Py_find_basename(const wchar_t *filename);
PyAPI_FUNC(wchar_t*) _Py_normpath(wchar_t *path, Py_ssize_t size);
extern wchar_t *_Py_normpath_and_size(wchar_t *path, Py_ssize_t size, Py_ssize_t *length);


// Macros to protect CRT calls against instant termination when passed an
// invalid parameter (bpo-23524). IPH stands for Invalid Parameter Handler.
// Usage:
//
//      _Py_BEGIN_SUPPRESS_IPH
//      ...
//      _Py_END_SUPPRESS_IPH
#if defined _MSC_VER && _MSC_VER >= 1900

#  include <stdlib.h>   // _set_thread_local_invalid_parameter_handler()

   extern _invalid_parameter_handler _Py_silent_invalid_parameter_handler;
#  define _Py_BEGIN_SUPPRESS_IPH \
    { _invalid_parameter_handler _Py_old_handler = \
      _set_thread_local_invalid_parameter_handler(_Py_silent_invalid_parameter_handler);
#  define _Py_END_SUPPRESS_IPH \
    _set_thread_local_invalid_parameter_handler(_Py_old_handler); }
#else
#  define _Py_BEGIN_SUPPRESS_IPH
#  define _Py_END_SUPPRESS_IPH
#endif /* _MSC_VER >= 1900 */

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_FILEUTILS_H */
