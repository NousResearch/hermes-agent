/* The PyMem_ family:  low-level memory allocation interfaces.
   See objimpl.h for the PyObject_ memory family.
*/

#ifndef Py_PYMEM_H
#define Py_PYMEM_H

#include "pyport.h"

#ifdef __cplusplus
extern "C" {
#endif

/* BEWARE:

   Each interface exports both functions and macros.  Extension modules should
   use the functions, to ensure binary compatibility across Python versions.
   Because the Python implementation is free to change internal details, and
   the macros may (or may not) expose details for speed, if you do use the
   macros you must recompile your extensions with each Python release.

   Never mix calls to PyMem_ with calls to the platform malloc/realloc/
   calloc/free.  For example, on Windows different DLLs may end up using
   different heaps, and if you use PyMem_Malloc you'll get the memory from the
   heap used by the Python DLL; it could be a disaster if you free()'ed that
   directly in your own extension.  Using PyMem_Free instead ensures Python
   can return the memory to the proper heap.  As another example, in
   a debug build (Py_DEBUG macro), Python wraps all calls to all PyMem_ and
   PyObject_ memory functions in special debugging wrappers that add additional
   debugging info to dynamic memory blocks.  The system routines have no idea
   what to do with that stuff, and the Python wrappers have no idea what to do
   with raw blocks obtained directly by the system routines then.

   The GIL must be held when using these APIs.
*/

/*
 * Raw memory interface
 * ====================
 */

/* Functions

   Functions supplying platform-independent semantics for malloc/realloc/
   free.  These functions make sure that allocating 0 bytes returns a distinct
   non-NULL pointer (whenever possible -- if we're flat out of memory, NULL
   may be returned), even if the platform malloc and realloc don't.
   Returned pointers must be checked for NULL explicitly.  No action is
   performed on failure (no exception is set, no warning is printed, etc).
*/

PyAPI_FUNC(void *) PyMem_Malloc(size_t size);
PyAPI_FUNC(void *) PyMem_Calloc(size_t nelem, size_t elsize);
PyAPI_FUNC(void *) PyMem_Realloc(void *ptr, size_t new_size);
PyAPI_FUNC(void) PyMem_Free(void *ptr);

/*
 * Type-oriented memory interface
 * ==============================
 *
 * Allocate memory for n objects of the given type.  Returns a new pointer
 * or NULL if the request was too large or memory allocation failed.  Use
 * these macros rather than doing the multiplication yourself so that proper
 * overflow checking is always done.
 */

#define PyMem_New(type, n) \
  ( ((size_t)(n) > PY_SSIZE_T_MAX / sizeof(type)) ? NULL :      \
        ( (type *) PyMem_Malloc((n) * sizeof(type)) ) )

/*
 * The value of (p) is always clobbered by this macro regardless of success.
 * The caller MUST check if (p) is NULL afterwards and deal with the memory
 * error if so.  This means the original value of (p) MUST be saved for the
 * caller's memory error handler to not lose track of it.
 */
#define PyMem_Resize(p, type, n) \
  ( (p) = ((size_t)(n) > PY_SSIZE_T_MAX / sizeof(type)) ? NULL :        \
        (type *) PyMem_Realloc((p), (n) * sizeof(type)) )


// Deprecated aliases only kept for backward compatibility.
// PyMem_Del and PyMem_DEL are defined with no parameter to be able to use
// them as function pointers (ex: dealloc = PyMem_Del).
#define PyMem_MALLOC(n)           PyMem_Malloc(n)
#define PyMem_NEW(type, n)        PyMem_New(type, n)
#define PyMem_REALLOC(p, n)       PyMem_Realloc(p, n)
#define PyMem_RESIZE(p, type, n)  PyMem_Resize(p, type, n)
#define PyMem_FREE(p)             PyMem_Free(p)
#define PyMem_Del                 PyMem_Free
#define PyMem_DEL                 PyMem_Free


#ifndef Py_LIMITED_API
#  define Py_CPYTHON_PYMEM_H
#  include "cpython/pymem.h"
#  undef Py_CPYTHON_PYMEM_H
#endif

#ifdef __cplusplus
}
#endif

#endif /* !Py_PYMEM_H */
