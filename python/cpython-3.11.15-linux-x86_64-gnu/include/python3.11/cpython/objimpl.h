#ifndef Py_CPYTHON_OBJIMPL_H
#  error "this header file must not be included directly"
#endif

#define _PyObject_SIZE(typeobj) ( (typeobj)->tp_basicsize )

/* _PyObject_VAR_SIZE returns the number of bytes (as size_t) allocated for a
   vrbl-size object with nitems items, exclusive of gc overhead (if any).  The
   value is rounded up to the closest multiple of sizeof(void *), in order to
   ensure that pointer fields at the end of the object are correctly aligned
   for the platform (this is of special importance for subclasses of, e.g.,
   str or int, so that pointers can be stored after the embedded data).

   Note that there's no memory wastage in doing this, as malloc has to
   return (at worst) pointer-aligned memory anyway.
*/
#if ((SIZEOF_VOID_P - 1) & SIZEOF_VOID_P) != 0
#   error "_PyObject_VAR_SIZE requires SIZEOF_VOID_P be a power of 2"
#endif

#define _PyObject_VAR_SIZE(typeobj, nitems)     \
    _Py_SIZE_ROUND_UP((typeobj)->tp_basicsize + \
        (nitems)*(typeobj)->tp_itemsize,        \
        SIZEOF_VOID_P)


/* This example code implements an object constructor with a custom
   allocator, where PyObject_New is inlined, and shows the important
   distinction between two steps (at least):
       1) the actual allocation of the object storage;
       2) the initialization of the Python specific fields
      in this storage with PyObject_{Init, InitVar}.

   PyObject *
   YourObject_New(...)
   {
       PyObject *op;

       op = (PyObject *) Your_Allocator(_PyObject_SIZE(YourTypeStruct));
       if (op == NULL) {
           return PyErr_NoMemory();
       }

       PyObject_Init(op, &YourTypeStruct);

       op->ob_field = value;
       ...
       return op;
   }

   Note that in C++, the use of the new operator usually implies that
   the 1st step is performed automatically for you, so in a C++ class
   constructor you would start directly with PyObject_Init/InitVar. */


typedef struct {
    /* user context passed as the first argument to the 2 functions */
    void *ctx;

    /* allocate an arena of size bytes */
    void* (*alloc) (void *ctx, size_t size);

    /* free an arena */
    void (*free) (void *ctx, void *ptr, size_t size);
} PyObjectArenaAllocator;

/* Get the arena allocator. */
PyAPI_FUNC(void) PyObject_GetArenaAllocator(PyObjectArenaAllocator *allocator);

/* Set the arena allocator. */
PyAPI_FUNC(void) PyObject_SetArenaAllocator(PyObjectArenaAllocator *allocator);


/* Test if an object implements the garbage collector protocol */
PyAPI_FUNC(int) PyObject_IS_GC(PyObject *obj);


/* Code built with Py_BUILD_CORE must include pycore_gc.h instead which
   defines a different _PyGC_FINALIZED() macro. */
#ifndef Py_BUILD_CORE
   // Kept for backward compatibility with Python 3.8
#  define _PyGC_FINALIZED(o) PyObject_GC_IsFinalized(o)
#endif


// Test if a type supports weak references
PyAPI_FUNC(int) PyType_SUPPORTS_WEAKREFS(PyTypeObject *type);

PyAPI_FUNC(PyObject **) PyObject_GET_WEAKREFS_LISTPTR(PyObject *op);
