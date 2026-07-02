#ifndef Py_INTERNAL_FLOATOBJECT_H
#define Py_INTERNAL_FLOATOBJECT_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif


/* runtime lifecycle */

extern void _PyFloat_InitState(PyInterpreterState *);
extern PyStatus _PyFloat_InitTypes(PyInterpreterState *);
extern void _PyFloat_Fini(PyInterpreterState *);
extern void _PyFloat_FiniType(PyInterpreterState *);


/* other API */

#ifndef WITH_FREELISTS
// without freelists
#  define PyFloat_MAXFREELIST 0
#endif

#ifndef PyFloat_MAXFREELIST
#  define PyFloat_MAXFREELIST   100
#endif

struct _Py_float_state {
#if PyFloat_MAXFREELIST > 0
    /* Special free list
       free_list is a singly-linked list of available PyFloatObjects,
       linked via abuse of their ob_type members. */
    int numfree;
    PyFloatObject *free_list;
#endif
};

void _PyFloat_ExactDealloc(PyObject *op);


PyAPI_FUNC(void) _PyFloat_DebugMallocStats(FILE* out);


/* Format the object based on the format_spec, as defined in PEP 3101
   (Advanced String Formatting). */
PyAPI_FUNC(int) _PyFloat_FormatAdvancedWriter(
    _PyUnicodeWriter *writer,
    PyObject *obj,
    PyObject *format_spec,
    Py_ssize_t start,
    Py_ssize_t end);

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_FLOATOBJECT_H */
