#ifndef Py_TRACEMALLOC_H
#define Py_TRACEMALLOC_H

#ifndef Py_LIMITED_API
/* Track an allocated memory block in the tracemalloc module.
   Return 0 on success, return -1 on error (failed to allocate memory to store
   the trace).

   Return -2 if tracemalloc is disabled.

   If memory block is already tracked, update the existing trace. */
PyAPI_FUNC(int) PyTraceMalloc_Track(
    unsigned int domain,
    uintptr_t ptr,
    size_t size);

/* Untrack an allocated memory block in the tracemalloc module.
   Do nothing if the block was not tracked.

   Return -2 if tracemalloc is disabled, otherwise return 0. */
PyAPI_FUNC(int) PyTraceMalloc_Untrack(
    unsigned int domain,
    uintptr_t ptr);

/* Get the traceback where a memory block was allocated.

   Return a tuple of (filename: str, lineno: int) tuples.

   Return None if the tracemalloc module is disabled or if the memory block
   is not tracked by tracemalloc.

   Raise an exception and return NULL on error. */
PyAPI_FUNC(PyObject*) _PyTraceMalloc_GetTraceback(
    unsigned int domain,
    uintptr_t ptr);
#endif

#endif /* !Py_TRACEMALLOC_H */
