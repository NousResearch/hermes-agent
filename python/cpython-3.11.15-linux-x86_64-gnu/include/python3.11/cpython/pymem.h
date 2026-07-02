#ifndef Py_CPYTHON_PYMEM_H
#  error "this header file must not be included directly"
#endif

PyAPI_FUNC(void *) PyMem_RawMalloc(size_t size);
PyAPI_FUNC(void *) PyMem_RawCalloc(size_t nelem, size_t elsize);
PyAPI_FUNC(void *) PyMem_RawRealloc(void *ptr, size_t new_size);
PyAPI_FUNC(void) PyMem_RawFree(void *ptr);

/* Try to get the allocators name set by _PyMem_SetupAllocators(). */
PyAPI_FUNC(const char*) _PyMem_GetCurrentAllocatorName(void);

/* strdup() using PyMem_RawMalloc() */
PyAPI_FUNC(char *) _PyMem_RawStrdup(const char *str);

/* strdup() using PyMem_Malloc() */
PyAPI_FUNC(char *) _PyMem_Strdup(const char *str);

/* wcsdup() using PyMem_RawMalloc() */
PyAPI_FUNC(wchar_t*) _PyMem_RawWcsdup(const wchar_t *str);


typedef enum {
    /* PyMem_RawMalloc(), PyMem_RawRealloc() and PyMem_RawFree() */
    PYMEM_DOMAIN_RAW,

    /* PyMem_Malloc(), PyMem_Realloc() and PyMem_Free() */
    PYMEM_DOMAIN_MEM,

    /* PyObject_Malloc(), PyObject_Realloc() and PyObject_Free() */
    PYMEM_DOMAIN_OBJ
} PyMemAllocatorDomain;

typedef enum {
    PYMEM_ALLOCATOR_NOT_SET = 0,
    PYMEM_ALLOCATOR_DEFAULT = 1,
    PYMEM_ALLOCATOR_DEBUG = 2,
    PYMEM_ALLOCATOR_MALLOC = 3,
    PYMEM_ALLOCATOR_MALLOC_DEBUG = 4,
#ifdef WITH_PYMALLOC
    PYMEM_ALLOCATOR_PYMALLOC = 5,
    PYMEM_ALLOCATOR_PYMALLOC_DEBUG = 6,
#endif
} PyMemAllocatorName;


typedef struct {
    /* user context passed as the first argument to the 4 functions */
    void *ctx;

    /* allocate a memory block */
    void* (*malloc) (void *ctx, size_t size);

    /* allocate a memory block initialized by zeros */
    void* (*calloc) (void *ctx, size_t nelem, size_t elsize);

    /* allocate or resize a memory block */
    void* (*realloc) (void *ctx, void *ptr, size_t new_size);

    /* release a memory block */
    void (*free) (void *ctx, void *ptr);
} PyMemAllocatorEx;

/* Get the memory block allocator of the specified domain. */
PyAPI_FUNC(void) PyMem_GetAllocator(PyMemAllocatorDomain domain,
                                    PyMemAllocatorEx *allocator);

/* Set the memory block allocator of the specified domain.

   The new allocator must return a distinct non-NULL pointer when requesting
   zero bytes.

   For the PYMEM_DOMAIN_RAW domain, the allocator must be thread-safe: the GIL
   is not held when the allocator is called.

   If the new allocator is not a hook (don't call the previous allocator), the
   PyMem_SetupDebugHooks() function must be called to reinstall the debug hooks
   on top on the new allocator. */
PyAPI_FUNC(void) PyMem_SetAllocator(PyMemAllocatorDomain domain,
                                    PyMemAllocatorEx *allocator);

/* Setup hooks to detect bugs in the following Python memory allocator
   functions:

   - PyMem_RawMalloc(), PyMem_RawRealloc(), PyMem_RawFree()
   - PyMem_Malloc(), PyMem_Realloc(), PyMem_Free()
   - PyObject_Malloc(), PyObject_Realloc() and PyObject_Free()

   Newly allocated memory is filled with the byte 0xCB, freed memory is filled
   with the byte 0xDB. Additional checks:

   - detect API violations, ex: PyObject_Free() called on a buffer allocated
     by PyMem_Malloc()
   - detect write before the start of the buffer (buffer underflow)
   - detect write after the end of the buffer (buffer overflow)

   The function does nothing if Python is not compiled is debug mode. */
PyAPI_FUNC(void) PyMem_SetupDebugHooks(void);
