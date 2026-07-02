#ifndef Py_INTERNAL_PYMEM_H
#define Py_INTERNAL_PYMEM_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#include "pymem.h"      // PyMemAllocatorName


/* Set the memory allocator of the specified domain to the default.
   Save the old allocator into *old_alloc if it's non-NULL.
   Return on success, or return -1 if the domain is unknown. */
PyAPI_FUNC(int) _PyMem_SetDefaultAllocator(
    PyMemAllocatorDomain domain,
    PyMemAllocatorEx *old_alloc);

/* Special bytes broadcast into debug memory blocks at appropriate times.
   Strings of these are unlikely to be valid addresses, floats, ints or
   7-bit ASCII.

   - PYMEM_CLEANBYTE: clean (newly allocated) memory
   - PYMEM_DEADBYTE dead (newly freed) memory
   - PYMEM_FORBIDDENBYTE: untouchable bytes at each end of a block

   Byte patterns 0xCB, 0xDB and 0xFB have been replaced with 0xCD, 0xDD and
   0xFD to use the same values than Windows CRT debug malloc() and free().
   If modified, _PyMem_IsPtrFreed() should be updated as well. */
#define PYMEM_CLEANBYTE      0xCD
#define PYMEM_DEADBYTE       0xDD
#define PYMEM_FORBIDDENBYTE  0xFD

/* Heuristic checking if a pointer value is newly allocated
   (uninitialized), newly freed or NULL (is equal to zero).

   The pointer is not dereferenced, only the pointer value is checked.

   The heuristic relies on the debug hooks on Python memory allocators which
   fills newly allocated memory with CLEANBYTE (0xCD) and newly freed memory
   with DEADBYTE (0xDD). Detect also "untouchable bytes" marked
   with FORBIDDENBYTE (0xFD). */
static inline int _PyMem_IsPtrFreed(const void *ptr)
{
    uintptr_t value = (uintptr_t)ptr;
#if SIZEOF_VOID_P == 8
    return (value == 0
            || value == (uintptr_t)0xCDCDCDCDCDCDCDCD
            || value == (uintptr_t)0xDDDDDDDDDDDDDDDD
            || value == (uintptr_t)0xFDFDFDFDFDFDFDFD);
#elif SIZEOF_VOID_P == 4
    return (value == 0
            || value == (uintptr_t)0xCDCDCDCD
            || value == (uintptr_t)0xDDDDDDDD
            || value == (uintptr_t)0xFDFDFDFD);
#else
#  error "unknown pointer size"
#endif
}

PyAPI_FUNC(int) _PyMem_GetAllocatorName(
    const char *name,
    PyMemAllocatorName *allocator);

/* Configure the Python memory allocators.
   Pass PYMEM_ALLOCATOR_DEFAULT to use default allocators.
   PYMEM_ALLOCATOR_NOT_SET does nothing. */
PyAPI_FUNC(int) _PyMem_SetupAllocators(PyMemAllocatorName allocator);

struct _PyTraceMalloc_Config {
    /* Module initialized?
       Variable protected by the GIL */
    enum {
        TRACEMALLOC_NOT_INITIALIZED,
        TRACEMALLOC_INITIALIZED,
        TRACEMALLOC_FINALIZED
    } initialized;

    /* Is tracemalloc tracing memory allocations?
       Variable protected by the GIL */
    int tracing;

    /* limit of the number of frames in a traceback, 1 by default.
       Variable protected by the GIL. */
    int max_nframe;
};

#define _PyTraceMalloc_Config_INIT \
    {.initialized = TRACEMALLOC_NOT_INITIALIZED, \
     .tracing = 0, \
     .max_nframe = 1}

PyAPI_DATA(struct _PyTraceMalloc_Config) _Py_tracemalloc_config;

/* Allocate memory directly from the O/S virtual memory system,
 * where supported. Otherwise fallback on malloc */
void *_PyObject_VirtualAlloc(size_t size);
void _PyObject_VirtualFree(void *, size_t size);

/* This function returns the number of allocated memory blocks, regardless of size */
PyAPI_FUNC(Py_ssize_t) _Py_GetAllocatedBlocks(void);

/* Macros */
#ifdef WITH_PYMALLOC
// Export the symbol for the 3rd party guppy3 project
PyAPI_FUNC(int) _PyObject_DebugMallocStats(FILE *out);
#endif

#ifdef __cplusplus
}
#endif
#endif  // !Py_INTERNAL_PYMEM_H
