#ifndef Py_INTERNAL_SYMTABLE_H
#define Py_INTERNAL_SYMTABLE_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

struct _mod;   // Type defined in pycore_ast.h

typedef enum _block_type { FunctionBlock, ClassBlock, ModuleBlock, AnnotationBlock }
    _Py_block_ty;

typedef enum _comprehension_type {
    NoComprehension = 0,
    ListComprehension = 1,
    DictComprehension = 2,
    SetComprehension = 3,
    GeneratorExpression = 4 } _Py_comprehension_ty;

struct _symtable_entry;

struct symtable {
    PyObject *st_filename;          /* name of file being compiled,
                                       decoded from the filesystem encoding */
    struct _symtable_entry *st_cur; /* current symbol table entry */
    struct _symtable_entry *st_top; /* symbol table entry for module */
    PyObject *st_blocks;            /* dict: map AST node addresses
                                     *       to symbol table entries */
    PyObject *st_stack;             /* list: stack of namespace info */
    PyObject *st_global;            /* borrowed ref to st_top->ste_symbols */
    int st_nblocks;                 /* number of blocks used. kept for
                                       consistency with the corresponding
                                       compiler structure */
    PyObject *st_private;           /* name of current class or NULL */
    PyFutureFeatures *st_future;    /* module's future features that affect
                                       the symbol table */
    int recursion_depth;            /* current recursion depth */
    int recursion_limit;            /* recursion limit */
};

typedef struct _symtable_entry {
    PyObject_HEAD
    PyObject *ste_id;        /* int: key in ste_table->st_blocks */
    PyObject *ste_symbols;   /* dict: variable names to flags */
    PyObject *ste_name;      /* string: name of current block */
    PyObject *ste_varnames;  /* list of function parameters */
    PyObject *ste_children;  /* list of child blocks */
    PyObject *ste_directives;/* locations of global and nonlocal statements */
    _Py_block_ty ste_type;   /* module, class or function */
    int ste_nested;      /* true if block is nested */
    unsigned ste_free : 1;        /* true if block has free variables */
    unsigned ste_child_free : 1;  /* true if a child block has free vars,
                                     including free refs to globals */
    unsigned ste_generator : 1;   /* true if namespace is a generator */
    unsigned ste_coroutine : 1;   /* true if namespace is a coroutine */
    _Py_comprehension_ty ste_comprehension;  /* Kind of comprehension (if any) */
    unsigned ste_varargs : 1;     /* true if block has varargs */
    unsigned ste_varkeywords : 1; /* true if block has varkeywords */
    unsigned ste_returns_value : 1;  /* true if namespace uses return with
                                        an argument */
    unsigned ste_needs_class_closure : 1; /* for class scopes, true if a
                                             closure over __class__
                                             should be created */
    unsigned ste_comp_iter_target : 1; /* true if visiting comprehension target */
    int ste_comp_iter_expr; /* non-zero if visiting a comprehension range expression */
    int ste_lineno;          /* first line of block */
    int ste_col_offset;      /* offset of first line of block */
    int ste_end_lineno;      /* end line of block */
    int ste_end_col_offset;  /* end offset of first line of block */
    int ste_opt_lineno;      /* lineno of last exec or import * */
    int ste_opt_col_offset;  /* offset of last exec or import * */
    struct symtable *ste_table;
} PySTEntryObject;

extern PyTypeObject PySTEntry_Type;

#define PySTEntry_Check(op) Py_IS_TYPE(op, &PySTEntry_Type)

extern long _PyST_GetSymbol(PySTEntryObject *, PyObject *);
extern int _PyST_GetScope(PySTEntryObject *, PyObject *);

extern struct symtable* _PySymtable_Build(
    struct _mod *mod,
    PyObject *filename,
    PyFutureFeatures *future);
PyAPI_FUNC(PySTEntryObject *) PySymtable_Lookup(struct symtable *, void *);

extern void _PySymtable_Free(struct symtable *);

/* Flags for def-use information */

#define DEF_GLOBAL 1           /* global stmt */
#define DEF_LOCAL 2            /* assignment in code block */
#define DEF_PARAM 2<<1         /* formal parameter */
#define DEF_NONLOCAL 2<<2      /* nonlocal stmt */
#define USE 2<<3               /* name is used */
#define DEF_FREE 2<<4          /* name used but not defined in nested block */
#define DEF_FREE_CLASS 2<<5    /* free variable from class's method */
#define DEF_IMPORT 2<<6        /* assignment occurred via import */
#define DEF_ANNOT 2<<7         /* this name is annotated */
#define DEF_COMP_ITER 2<<8     /* this name is a comprehension iteration variable */

#define DEF_BOUND (DEF_LOCAL | DEF_PARAM | DEF_IMPORT)

/* GLOBAL_EXPLICIT and GLOBAL_IMPLICIT are used internally by the symbol
   table.  GLOBAL is returned from PyST_GetScope() for either of them.
   It is stored in ste_symbols at bits 12-15.
*/
#define SCOPE_OFFSET 11
#define SCOPE_MASK (DEF_GLOBAL | DEF_LOCAL | DEF_PARAM | DEF_NONLOCAL)

#define LOCAL 1
#define GLOBAL_EXPLICIT 2
#define GLOBAL_IMPLICIT 3
#define FREE 4
#define CELL 5

#define GENERATOR 1
#define GENERATOR_EXPRESSION 2

// Used by symtablemodule.c
extern struct symtable* _Py_SymtableStringObjectFlags(
    const char *str,
    PyObject *filename,
    int start,
    PyCompilerFlags *flags);

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_SYMTABLE_H */
