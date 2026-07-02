/* The PyObject_ memory family:  high-level object memory interfaces.
   See pymem.h for the low-level PyMem_ family.
*/

#ifndef Py_OBJIMPL_H
#define Py_OBJIMPL_H

#include "pymem.h"

#ifdef __cplusplus
extern "C" {
#endif

/* BEWARE:

   Each interface exports both functions and macros.  Extension modules should
   use the functions, to ensure binary compatibility across Python versions.
   Because the Python implementation is free to change internal details, and
   the macros may (or may not) expose details for speed, if you do use the
   macros you must recompile your extensions with each Python release.

   Never mix calls to PyObject_ memory functions with calls to the platform
   malloc/realloc/ calloc/free, or with calls to PyMem_.
*/

/*
Functions and macros for modules that implement new object types.

 - PyObject_New(type, typeobj) allocates memory for a new object of the given
   type, and initializes part of it.  'type' must be the C structure type used
   to represent the object, and 'typeobj' the address of the corresponding
   type object.  Reference count and type pointer are filled in; the rest of
   the bytes of the object are *undefined*!  The resulting expression type is
   'type *'.  The size of the object is determined by the tp_basicsize field
   of the type object.

 - PyObject_NewVar(type, typeobj, n) is similar but allocates a variable-size
   object with room for n items.  In addition to the refcount and type pointer
   fields, this also fills in the ob_size field.

 - PyObject_Free(op) releases the memory allocated for an object.  It does not
   run a destructor -- it only frees the memory.  PyObject_Free is identical.

 - PyObject_Init(op, typeobj) and PyObject_InitVar(op, typeobj, n) don't
   allocate memory.  Instead of a 'type' parameter, they take a pointer to a
   new object (allocated by an arbitrary allocator), and initialize its object
   header fields.

Note that objects created with PyObject_{New, NewVar} are allocated using the
specialized Python allocator (implemented in obmalloc.c), if WITH_PYMALLOC is
enabled.  In addition, a special debugging allocator is used if Py_DEBUG
macro is also defined.

In case a specific form of memory management is needed (for example, if you
must use the platform malloc heap(s), or shared memory, or C++ local storage or
operator new), you must first allocate the object with your custom allocator,
then pass its pointer to PyObject_{Init, InitVar} for filling in its Python-
specific fields:  reference count, type pointer, possibly others.  You should
be aware that Python has no control over these objects because they don't
cooperate with the Python memory manager.  Such objects may not be eligible
for automatic garbage collection and you have to make sure that they are
released accordingly whenever their destructor gets called (cf. the specific
form of memory management you're using).

Unless you have specific memory management requirements, use
PyObject_{New, NewVar, Del}.
*/

/*
 * Raw object memory interface
 * ===========================
 */

/* Functions to call the same malloc/realloc/free as used by Python's
   object allocator.  If WITH_PYMALLOC is enabled, these may differ from
   the platform malloc/realloc/free.  The Python object allocator is
   designed for fast, cache-conscious allocation of many "small" objects,
   and with low hidden memory overhead.

   PyObject_Malloc(0) returns a unique non-NULL pointer if possible.

   PyObject_Realloc(NULL, n) acts like PyObject_Malloc(n).
   PyObject_Realloc(p != NULL, 0) does not return  NULL, or free the memory
   at p.

   Returned pointers must be checked for NULL explicitly; no action is
   performed on failure other than to return NULL (no warning it printed, no
   exception is set, etc).

   For allocating objects, use PyObject_{New, NewVar} instead whenever
   possible.  The PyObject_{Malloc, Realloc, Free} family is exposed
   so that you can exploit Python's small-block allocator for non-object
   uses.  If you must use these routines to allocate object memory, make sure
   the object gets initialized via PyObject_{Init, InitVar} after obtaining
   the raw memory.
*/
PyAPI_FUNC(void *) PyObject_Malloc(size_t size);
#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x03050000
PyAPI_FUNC(void *) PyObject_Calloc(size_t nelem, size_t elsize);
#endif
PyAPI_FUNC(void *) PyObject_Realloc(void *ptr, size_t new_size);
PyAPI_FUNC(void) PyObject_Free(void *ptr);


// Deprecated aliases only kept for backward compatibility.
// PyObject_Del and PyObject_DEL are defined with no parameter to be able to
// use them as function pointers (ex: tp_free = PyObject_Del).
#define PyObject_MALLOC         PyObject_Malloc
#define PyObject_REALLOC        PyObject_Realloc
#define PyObject_FREE           PyObject_Free
#define PyObject_Del            PyObject_Free
#define PyObject_DEL            PyObject_Free


/*
 * Generic object allocator interface
 * ==================================
 */

/* Functions */
PyAPI_FUNC(PyObject *) PyObject_Init(PyObject *, PyTypeObject *);
PyAPI_FUNC(PyVarObject *) PyObject_InitVar(PyVarObject *,
                                           PyTypeObject *, Py_ssize_t);

#define PyObject_INIT(op, typeobj) \
    PyObject_Init(_PyObject_CAST(op), (typeobj))
#define PyObject_INIT_VAR(op, typeobj, size) \
    PyObject_InitVar(_PyVarObject_CAST(op), (typeobj), (size))


PyAPI_FUNC(PyObject *) _PyObject_New(PyTypeObject *);
PyAPI_FUNC(PyVarObject *) _PyObject_NewVar(PyTypeObject *, Py_ssize_t);

#define PyObject_New(type, typeobj) ((type *)_PyObject_New(typeobj))

// Alias to PyObject_New(). In Python 3.8, PyObject_NEW() called directly
// PyObject_MALLOC() with _PyObject_SIZE().
#define PyObject_NEW(type, typeobj) PyObject_New(type, typeobj)

#define PyObject_NewVar(type, typeobj, n) \
                ( (type *) _PyObject_NewVar((typeobj), (n)) )

// Alias to PyObject_NewVar(). In Python 3.8, PyObject_NEW_VAR() called
// directly PyObject_MALLOC() with _PyObject_VAR_SIZE().
#define PyObject_NEW_VAR(type, typeobj, n) PyObject_NewVar(type, typeobj, n)


/*
 * Garbage Collection Support
 * ==========================
 */

/* C equivalent of gc.collect(). */
PyAPI_FUNC(Py_ssize_t) PyGC_Collect(void);
/* C API for controlling the state of the garbage collector */
PyAPI_FUNC(int) PyGC_Enable(void);
PyAPI_FUNC(int) PyGC_Disable(void);
PyAPI_FUNC(int) PyGC_IsEnabled(void);

/* Test if a type has a GC head */
#define PyType_IS_GC(t) PyType_HasFeature((t), Py_TPFLAGS_HAVE_GC)

PyAPI_FUNC(PyVarObject *) _PyObject_GC_Resize(PyVarObject *, Py_ssize_t);
#define PyObject_GC_Resize(type, op, n) \
                ( (type *) _PyObject_GC_Resize(_PyVarObject_CAST(op), (n)) )



PyAPI_FUNC(PyObject *) _PyObject_GC_New(PyTypeObject *);
PyAPI_FUNC(PyVarObject *) _PyObject_GC_NewVar(PyTypeObject *, Py_ssize_t);

/* Tell the GC to track this object.
 *
 * See also private _PyObject_GC_TRACK() macro. */
PyAPI_FUNC(void) PyObject_GC_Track(void *);

/* Tell the GC to stop tracking this object.
 *
 * See also private _PyObject_GC_UNTRACK() macro. */
PyAPI_FUNC(void) PyObject_GC_UnTrack(void *);

PyAPI_FUNC(void) PyObject_GC_Del(void *);

#define PyObject_GC_New(type, typeobj) \
    _Py_CAST(type*, _PyObject_GC_New(typeobj))
#define PyObject_GC_NewVar(type, typeobj, n) \
    _Py_CAST(type*, _PyObject_GC_NewVar((typeobj), (n)))

PyAPI_FUNC(int) PyObject_GC_IsTracked(PyObject *);
PyAPI_FUNC(int) PyObject_GC_IsFinalized(PyObject *);

/* Utility macro to help write tp_traverse functions.
 * To use this macro, the tp_traverse function must name its arguments
 * "visit" and "arg".  This is intended to keep tp_traverse functions
 * looking as much alike as possible.
 */
#define Py_VISIT(op)                                                    \
    do {                                                                \
        if (op) {                                                       \
            int vret = visit(_PyObject_CAST(op), arg);                  \
            if (vret)                                                   \
                return vret;                                            \
        }                                                               \
    } while (0)

#ifndef Py_LIMITED_API
#  define Py_CPYTHON_OBJIMPL_H
#  include "cpython/objimpl.h"
#  undef Py_CPYTHON_OBJIMPL_H
#endif

#ifdef __cplusplus
}
#endif
#endif /* !Py_OBJIMPL_H */
