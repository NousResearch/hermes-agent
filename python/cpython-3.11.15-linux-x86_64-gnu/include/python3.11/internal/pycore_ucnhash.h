/* Unicode name database interface */
#ifndef Py_INTERNAL_UCNHASH_H
#define Py_INTERNAL_UCNHASH_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

/* revised ucnhash CAPI interface (exported through a "wrapper") */

#define PyUnicodeData_CAPSULE_NAME "unicodedata._ucnhash_CAPI"

typedef struct {

    /* Get name for a given character code.
       Returns non-zero if success, zero if not.
       Does not set Python exceptions. */
    int (*getname)(Py_UCS4 code, char* buffer, int buflen,
                   int with_alias_and_seq);

    /* Get character code for a given name.
       Same error handling as for getname(). */
    int (*getcode)(const char* name, int namelen, Py_UCS4* code,
                   int with_named_seq);

} _PyUnicode_Name_CAPI;

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_UCNHASH_H */
