#ifndef Py_INTERNAL_BYTESOBJECT_H
#define Py_INTERNAL_BYTESOBJECT_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif


/* runtime lifecycle */

extern PyStatus _PyBytes_InitTypes(PyInterpreterState *);


/* Substring Search.

   Returns the index of the first occurrence of
   a substring ("needle") in a larger text ("haystack").
   If the needle is not found, return -1.
   If the needle is found, add offset to the index.
*/

PyAPI_FUNC(Py_ssize_t)
_PyBytes_Find(const char *haystack, Py_ssize_t len_haystack,
              const char *needle, Py_ssize_t len_needle,
              Py_ssize_t offset);

/* Same as above, but search right-to-left */
PyAPI_FUNC(Py_ssize_t)
_PyBytes_ReverseFind(const char *haystack, Py_ssize_t len_haystack,
                     const char *needle, Py_ssize_t len_needle,
                     Py_ssize_t offset);


/** Helper function to implement the repeat and inplace repeat methods on a buffer
 *
 * len_dest is assumed to be an integer multiple of len_src.
 * If src equals dest, then assume the operation is inplace.
 *
 * This method repeately doubles the number of bytes copied to reduce
 * the number of invocations of memcpy.
 */
PyAPI_FUNC(void)
_PyBytes_Repeat(char* dest, Py_ssize_t len_dest,
    const char* src, Py_ssize_t len_src);

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_BYTESOBJECT_H */
